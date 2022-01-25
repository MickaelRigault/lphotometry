#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pymage
import pandas
import warnings
from scipy import stats



from astropy import units, constants
from pymage import query


from . import target
from . import dataquery
GALEXQUERY = query.GALEXQuery()

# ======================= #
#                         #
#                         #
#  GENERAL FUNCTIONS      #
#                         #
#                         #
# ======================= #


# ======================= #
#                         #
#                         #
#  CLASSES                #
#                         #
#                         #
# ======================= #
class _Photomatize_( target._TargetHandler_ ):
    
    def __init__(self, target=None, load=True, radius=None, runits="kpc", **kwargs):
        """ """
        #
        _ = super().__init__(target=target)
        #
        if radius is not None:
            if not load:
                warnings.warn("load set to True in order to measure the photometry")
            load = True

        if load:
            self.load_data(**kwargs)
            
        if radius is not None:
            self.measure_photometry(radius, runits="kpc")
            
    # ============= #
    #  Methods      #
    # ============= #
    def measure_photometry(self, radius, runits="kpc", verbose=False):
        """ """
        raise NotImplementedError("")

    def _local_photometry_setup_(self, radius, runits="kpc"):
        """ """
        if runits !="kpc":
            print("WARNING: self.surface not in kpc2")
            
        self.surface = np.pi* (radius*getattr(units,runits))**2
        return self.to_arcsec(radius, runits="kpc")
    
    def to_arcsec(self, radius, runits="kpc"):
        """ """
        if runits == "kpc":
            radius = radius * self.target.arcsec_per_kpc.value
            runits = "arcsec"
            self._radius_arcsec = radius
        else:
            self._radius_arcsec = radius*getattr(units,runits).to("arcsec")
            
        return self._radius_arcsec, "arcsec"
        
    # --------- #
    # LOAD/SET  #
    # --------- #
    def load_data(self, forcedl=False, **kwargs):
        """ Loads the galex instrument corresponding to self.target 
        
        **kwargs goes to dataquery.get_instruments()
        """
        data = dataquery.get_instrument(self.INSTNAME, self.target,
                                        forcedl=forcedl, **kwargs)
        for label, inst_ in data.items():
            self.set_instrument(inst_, label)
            
    def set_instrument(self, instrument, label=None, overwrite=False, checkbandname=True):
        """ """
        if not hasattr(self, "_instruments"):
            self._instruments = {}
        
        if instrument is None:
            if label is None:
                raise ValueError("Instrument is not, and label is not given")
            self._instruments[label] = instrument
            return

        # - instrument is not None
        if label is None:
            label = instrument.bandname
            
        if instrument.bandname not in self.BANDS.keys() and checkbandname:
            raise ValueError("The given instrument has bandaname=%s and should not be set. Accepted bandnames are: "%instrument.bandname+\
                                 ", ".join(list(self.BANDS.keys())))
        


        from astrobject import get_target
        target = get_target(name=self.target.name,
                                ra=self.target.radec[0],
                                dec=self.target.radec[1],
                                zcmb=self.target.zcmb)
        instrument.set_target( target )
            
        if label in self._instruments and not overwrite:
            raise ValueError("%s already set and overwrite is False")
        
        self._instruments[label] = instrument
        
    # --------- #
    #  GETTER   #
    # --------- #        
    def get_photometry(self, band, radius=None, runits='kpc',
                           mag=True, full=False, error_floor="default"):
        """ Get the photometry as stored in the self.data 
        note: you need to have run measure_photomtery() 

        Parameters
        ----------
        band: [string]
            name of the band as stored in self.data["bandname"] 

        radius: [float or None]
            provide the radius of the local photometry.
            If None, this will use the current data measured with self.measure_photometry().
            If given, this will re measure the photometry (and update self.data)

        error_floor: [float] -optional
            error floor to add to the data. This error will be added quadradively.

            - if mag=True: error_floor is understood in magnitude
                           -> e_mag = np.sqrt(e_mag_orig**2 + error_floor**2)
            - if mag=False: (so flux) error_floor is understood as fraction of flux
                           -> efloor = flux*error_floor
                           -> e_flux = np.sqrt( e_flux_orig**2 + efloor**2)

            If error_floor is None: this is ignored.
            If error_floor is default, that defined in self.BANDS[band][error_floor] is used
        """
        
        print(" = missing MW CORR =")
        
        if not self.has_data():
            if radius is None:
                raise ValueError("No local_photometry measured ('self.measure_photometry()') and no radius given")
            else:
                self.measure_photometry(radius, runits=runits)
                
        if len(self.data)==0:
            warnings.warn("no data (no instrument?) Nan returned.")
            return np.NaN, np.NaN
        
        db = self.data[self.data["bandname"]==band]
        if db.size == 0:
            warnings.warn(f"no data associated to {band}")
            return None, None
            
        key = "mag" if mag else "flux"
        if not full:
            if len(db[key])>1:
                mean_ = np.average(db[key],weights=1/db[key+".err"]**2)
                mean_err = np.sqrt(1/np.sum(1/db[key+".err"]**2))
            
            else:
                mean_,mean_err = np.asarray(db[[key,key+".err"]].values[0], dtype="float")
        else:
            mean_,mean_err = db[[key,key+".err"]]

        if error_floor == "default":
            if not hasattr(self,"BANDS"):
                warnings.warn("self.BANDS is not defined. No default error_floor")
                error_floor = None
                
            error_floor = self.BANDS.get(band, {}).get("error_floor", "_backup_None")
            if error_floor == "_backup_None":
                warnings.warn(r"no error_floor set for {band}.")
                error_floor = None

        if error_floor is not None:
            if not mag:
                error_floor = mean_*error_floor
            mean_err = np.sqrt(mean_err**2 + error_floor**2)

        return mean_,mean_err
    
    def get_instrument(self, bandname):
        """ get a copy on the instrument """
        img = self.instruments[bandname]
        if img is None:
            return None
        return img.copy()


    def get_localdata(self, band, radius, runits="kpc", **kwargs):
        """ get the image center on the target coordinate (radius = half the square size) """
        instru = self.get_instrument(band, **kwargs)
        if instru is None:
            return [np.NaN, np.NaN], None
        size_pixels = radius * instru.units_to_pixels( runits ).value
        xpix, ypix = instru.coords_to_pixel(instru.target.ra, instru.target.dec)
        ymin,ymax,xmin,xmax = int(np.rint(ypix-size_pixels)),int(np.rint(ypix+size_pixels)), int(np.rint(xpix-size_pixels)),int(np.rint(xpix+size_pixels))
        return (xpix-xmin,ypix-ymin), instru.data[ymin:ymax,xmin:xmax]
        


    def show_stamps(self, ax=None, savefile=None, figsize=None, colorbase="white",
                        radius_kpc=10,
                        linecolor="w", textcolor="0.9", vmin="1",vmax="99",
                        stretching="arcsinh", **kwargs):
        """ 

        Parameters
        ----------

        colorbase: [string] -optional-
            colormaps used for displaying the stamps
            - white: [matplotlib's Blues, Reds etc]
            - color: reversed of 'white' [Blues_r, Reds_r etc.]
            - black [cmasher colors]


        stretching: [string or None] -optional-
            How the data are transform before entering imshow.
            This must be a numpy function.
            (so imshow( np.`stretching`(data) ))
            The vmin and vmax options are applied on the stretched data.
            If None, no stretching applied.
        """
        import matplotlib.pyplot as mpl
        from matplotlib.patches import Circle
        from .tools import parse_vmin_vmax
        if colorbase == "black":
            import cmasher as cmr
            CMAPS = {"fuv":mpl.get_cmap("cmr.cosmic"),
                     "nuv":mpl.get_cmap("cmr.freeze"),
                     #
                    "g":mpl.get_cmap("cmr.jungle"),
                    "r":mpl.get_cmap("cmr.flamingo"),
                    "i":mpl.get_cmap("cmr.amber")
                    }
        else:
            CMAPS_full = {"white":{  "fuv":mpl.get_cmap("Purples"),
                                 "nuv":mpl.get_cmap("Blues"),
                                 #
                                 "g":mpl.get_cmap("Greens"),
                                 "r":mpl.get_cmap("Reds"),
                                 "i":mpl.get_cmap("Oranges"),
                                },
                         "color":{  "fuv":mpl.get_cmap("Purples_r"),
                                 "nuv":mpl.get_cmap("Blues_r"),
                                 #
                                 "g":mpl.get_cmap("Greens_r"),
                                 "r":mpl.get_cmap("Reds_r"),
                                 "i":mpl.get_cmap("Oranges_r"),
                                }
                          }
            CMAPS = CMAPS_full[colorbase]                 

        #####
        instrumentnames = [k for k in self.instruments.keys() if "_" not in k]
        ninstruments = len(instrumentnames)
        if figsize is None:
            figsize = [2*ninstruments,2]

        if ax is None:
            fig = mpl.figure( figsize=figsize )
            width, spanx = 0.8/ninstruments, 0.02
            fullwidth = 0.8 + spanx*(ninstruments-1)
            edges = 1-fullwidth

            ax = []
            for i,band  in enumerate(instrumentnames):
                ax.append(fig.add_axes([edges/2+i*(width+spanx),0.15,width,0.75]))
        else:
            fig = ax[0].figure
            
        prop = {**dict( origin="lower"), **kwargs}

        for i, band in enumerate(instrumentnames):
            ax_ = ax[i]
            
            if radius_kpc is not None:
                centroid, data = self.get_localdata(band, radius_kpc, "kpc")
            else:
                instru = self.get_instrument(band)
                data  = instru.data
                centroid = instru.coords_to_pixel(*self.target.radec)

            # - No Data
            if data is None:
                ax_.text(0.5,0.5, f"no {band} band", va="center",ha="center", 
                          transform=ax_.transAxes, color=CMAPS[band](0.5), fontsize="medium")
                continue
            
            # - Data                
            if stretching is not None:
                data = getattr(np,stretching)(data)
            vmin_, vmax_ = parse_vmin_vmax(data, vmin, vmax)
            ax_.imshow( data, vmin=vmin_, vmax=vmax_, cmap=CMAPS[band], **prop)

            arcsec_kpc = self.get_instrument(band).units_to_pixels( "kpc" ).value
            ax_.add_patch( 
                    Circle(centroid, radius = 1*arcsec_kpc,
                      facecolor="None", edgecolor=linecolor, lw=1.5)
                    )
            ax_.add_patch( 
                    Circle(centroid, radius = 3*arcsec_kpc,
                      facecolor="None", edgecolor=linecolor, lw=0.5, ls="--")
                    )
            ax_.text(0.02,0.98, f"{band}", va="top",ha="left", 
                          transform=ax_.transAxes, color=textcolor, fontsize="small")

        [ax_.set_yticks([]) for ax_ in ax]
        [ax_.set_xticks([]) for ax_ in ax]

        ax[0].set_ylabel(self.target.name)

        if savefile is not None:
            fig.savefig(savefile)
            
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def data(self):
        """ """
        if not self.has_data():
            raise AttributeError("No data measured. run measure_photometry()")
        return self._data

    def has_data(self):
        """ """
        return hasattr(self,"_data")

    @property
    def photopoints(self):
        """ """
        if not hasattr(self,"_photopoints"):
            raise AttributeError("No data measured. run measure_photometry()")
        return self._photopoints

    
    @property
    def bandnames(self):
        """ """
        return np.asarray([]) if len(self.data)==0 else np.unique(self.data['bandname'])
    
    @property
    def instruments(self):
        """ """
        if not hasattr(self, "_instruments"):
            self._instruments = {}
        return self._instruments

    
#               #
#   OPTICAL     #
#               #

class PS1LocalMass( _Photomatize_ ):

    BANDS = {"ps1.g":{"lbda":4866.457871, "error_floor":0.03},
             "ps1.r":{"lbda":6214.623038, "error_floor":0.03},
             "ps1.i":{"lbda":7544.570357, "error_floor":0.03}
            }

    INSTNAME = "panstarrs"
    
    def measure_photometry(self, radius, runits="kpc", verbose=False):
        """ """
        radius, runits = self._local_photometry_setup_(radius, runits=runits)
        self._photopoints = {}
        dataout = {}
        for bandname, g_ in self.instruments.items():
            if verbose:
                print(g_.filename)
            if g_ is None:
                dataout[bandname] = {k:None for k in ["flux","flux.err","mag","mag.errasym","mag.err",
                                                      "target.x","target.y"]}
                dataout[bandname]["bandname"] = bandname
                continue
            ppc = g_.get_target_photopoint(radius, runits=runits, on="data")
            self._photopoints[bandname] = ppc
            gout = ppc.data.copy()
            gout["flux"] = ppc.flux
            gout["flux.err"] = np.sqrt(ppc.var)
            gout["mag"] = ppc.mag
            gout["mag.errasym"] = ppc.mag_err
            gout["mag.err"] = np.mean(ppc.mag_err)
            gout["target.x"],gout["target.y"] = g_.coords_to_pixel(*self.target.radec)
 
            dataout[bandname] = gout
            
        self._data = pandas.DataFrame(dataout).T
        self._mass_estimator = None # clearit
        return self.data

    # -------- #
    #  GETTER  #
    # -------- #
    def get_mass_estimator(self, radius=None, runits="kpc"):
        """ """
        if radius is not None:
            self.measure_photometry(radius=radius,runits=runits)

        from .mass import MassEstimator
        gmag = self.get_photometry("ps1.g", mag=True)
        imag = self.get_photometry("ps1.i", mag=True)
        return MassEstimator(gmag=gmag, imag=imag, distmpc=self.target.distmpc)

    def get_mass(self, radius=None, runits="kpc", refsize=None, **kwargs):
        """ """
        if radius is not None:
            self.measure_photometry(radius=radius,runits=runits)

        offset=self._get_surface_offset_(refsize)
        return self.mass_estimator.get_mass(offset=offset, **kwargs)
        
    def get_backup_mass(self):
        """ Assuming the mean and the std of the local mass distribution from Rigault et al. 2018"""
        return 8.06, 0.58 #from Rigault et al. 2018 (SNfactory data)

    def _get_surface_offset_(self, refsize):
        """ """
        if refsize is None:
            return refsize
        return np.log10( self.surface.value/(np.pi*refsize**2) )
    
    # -------- #
    # PLOTTERS #
    # -------- #
    def show_mass(self, ax=None, savefile=None, refsize=None,
                      clear_axes=["left","right","top"],
                      color="purple", color_nodust=None, set_label=True, r13_color="k",
                      **kwargs):
        """ """
        offset=self._get_surface_offset_(refsize)
        return self.mass_estimator.show_mass(offset=offset,
                                             ax=ax, savefile=savefile, 
                                             clear_axes=clear_axes,
                                             color=color, color_nodust=color_nodust,
                                             set_label=set_label, r13_color=r13_color)
        

    def show_gicolor(self, ax=None, savefile=None,
                      color_prior="0.2", color_data="C0",
                      color_inferred="C1", **kwargs):
        """ """
        return self.mass_estimator.show_gicolor( ax=ax, savefile=savefile,
                                                 color_prior=color_prior, color_data=color_data,
                                                 color_inferred=color_inferred,
                                                 **kwargs)
    
    # ================== #
    #   Properties       #
    # ================== #
    def has_instruments(self):
        """ Check if any of the instrument is not None """
        return np.any([img_ is not None for img_ in self.instruments.values()])

    @property
    def mass_estimator(self):
        """ """
        if not hasattr(self, "_mass_estimator") or self._mass_estimator is None:
            self._mass_estimator = self.get_mass_estimator()
        return self._mass_estimator
            
#               #
#   GALEX       #
#               #
class UVLocalSFR( _Photomatize_ ):
    """ """
    BANDS = {"nuv":{"lbda":2315.66, "error_floor":None},
             "fuv":{"lbda":1538.62, "error_floor":None},
            }
        
    INSTNAME = "galex"
    
    # ============= #
    #  Methods      #
    # ============= #
    def measure_photometry(self, radius, runits="kpc", verbose=False):
        """ """
        radius, runits = self._local_photometry_setup_(radius, runits=runits)
        dataout = {}
        self._photopoints = {}
        for label_, g_ in self.instruments.items():
            if verbose:
                print(g_.filename)
            ppc = g_.get_target_photopoint(radius, runits=runits)
            self._photopoints[label_] = ppc            
            gout = ppc.data.copy()
            gout["cps"] = ppc.cps
            gout["cps.err"] = ppc.cps_err
            gout["flux"] = ppc.flux
            gout["flux.err"] = np.sqrt(ppc.var)
            gout["mag"] = ppc.mag
            gout["mag.errasym"] = ppc.mag_err
            gout["mag.err"] = np.mean(ppc.mag_err)
            gout["target.x"],gout["target.y"] = g_.coords_to_pixel(*self.target.radec)
            gout["filename"] = g_.header["OBJECT"]
            dataout[label_] = gout
            
        self._data = pandas.DataFrame(dataout).T
        self._sfr_estimator = None
        return self.data

    # ------ #
    # GETTER #
    # ------ #
    def get_instrument(self, band, which="deepest"):
        """ """
        band_data = self.data[self.data["bandname"]==band]
        if which in ["deepest","longest"]:
            selected_filename = band_data.sort_values("exptime", ascending=False).index[0]
        elif which in ["shallowest", "shortest", "faintest"]:
            selected_filename = band_data.sort_values("exptime", ascending=True).index[0]
        else:
            raise ValueError("which can only be deepest or shallowest")
        
        return self.instruments[selected_filename]

    # -------- #
    #  GETTER  #
    # -------- #
    def get_sfr_estimator(self, radius=None, runits="kpc"):
        """ """
        if radius is not None:
            self.measure_photometry(radius=radius,runits=runits)

        from .sfr import GalexSFREstimator
        nuv = self.get_photometry("nuv", mag=True)
        fuv = self.get_photometry("fuv", mag=True)
        return GalexSFREstimator(fuv=fuv, nuv=nuv, distmpc=self.target.distmpc)

    def get_sfr(self, radius=None, runits="kpc", **kwargs):
        """ """
        if radius is not None:
            self.measure_photometry(radius=radius,runits=runits)

        return self.sfr_estimator.get_sfr(surface=self.surface.value, **kwargs)

    def get_nuvbackup_sfr(self, radius=None, runits="kpc", **kwargs):
        """ """
        if radius is not None:
            self.measure_photometry(radius=radius,runits=runits)

        return self.sfr_estimator.get_nuvbackup_sfr(surface=self.surface.value, **kwargs)
    
    def show_sfr(self, ax=None, savefile=None,
                       clear_axes=["left","right","top"],
                       color="purple", color_nodust=None,
                     set_label=True, r13_color="k", **kwargs):
        """ """
        surface = self.surface.value
        return self.sfr_estimator.show_sfr( ax=ax, savefile=savefile, surface=surface,
                                            clear_axes=clear_axes,
                                            color=color, color_nodust=color_nodust,
                                            set_label=set_label, r13_color=r13_color, **kwargs)
    
    def show_afuv(self, ax=None, savefile=None,
                        color_prior="0.2", color_data="C0",
                        color_posterior="C1", color_inferred=None,
                        set_legend=True, set_label=True,
                      **kwargs):
        """ """
        return self.sfr_estimator.show_afuv(ax=ax, savefile=savefile,
                                            color_prior=color_prior, color_data=color_data,
                                            color_posterior=color_posterior, color_inferred=color_inferred,
                                            set_legend=set_legend, set_label=set_label,
                                            **kwargs)
    
    # ================== #
    #   Properties       #
    # ================== #
    def has_instruments(self):
        """ Check if any of the instrument is not None """
        return np.any([img_ is not None for img_ in self.instruments.values()])

    @property
    def sfr_estimator(self):
        """ """
        if not hasattr(self, "_sfr_estimator") or self._sfr_estimator is None:
            self._sfr_estimator = self.get_sfr_estimator()
        return self._sfr_estimator
