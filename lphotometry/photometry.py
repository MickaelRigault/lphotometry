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

def salim_afuv(uvcolor, use_sfcurve=True, squeeze=True):
    """ uvcolor = FUV-NUV restframe see Salim et al. 2007 """
    coef, offset,cut, backup = (3.32, 0.22, 0.95, 3.37) if use_sfcurve else \
                               (2.99, 0.27, 0.90, 2.96)
                               
    uvc = np.atleast_1d(uvcolor)
    afuv_base = coef*uvc + offset
    afuv_base[uvc>cut] = backup
    
    return np.squeeze(afuv_base) if squeeze else afuv_base
    

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
        if label is None:
            label = instrument.bandname
            
        if instrument.bandname not in self.BANDS.keys() and checkbandname:
            raise ValueError("The given instrument has bandaname=%s and should not be set. Accepted bandnames are: "%instrument.bandname+\
                                 ", ".join(list(self.BANDS.keys())))
        
        if not hasattr(self, "_instruments"):
            self._instruments = {}


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
                           mag=True, full=False):
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
        """
        if not self.has_data():
            if radius is None:
                raise ValueError("No local_photometry measured ('self.measure_photometry()') and no radius given")
            else:
                self.measure_photometry(radius, runits=runits)
                
        if len(self.data)==0:
            warnings.warn("no data (no instrument?) Nan returned.")
            return np.NaN, np.NaN
        
        db = self.data[self.data["bandname"]==band]
        key = "mag" if mag else "flux"
        if not full:
            if len(db[key])>1:
                mean_ = np.average(db[key],weights=1/db[key+".err"]**2)
                mean_err = np.sqrt(1/np.sum(1/db[key+".err"]**2))
                return mean_,mean_err
            
            return np.asarray(db[[key,key+".err"]].values[0], dtype="float")
        
        return db[[key,key+".err"]]

    def get_instrument(self, bandname):
        """ get a copy on the instrument """
        return self.instruments[bandname].copy()


    def get_localdata(self, band, radius, runits="kpc", **kwargs):
        """ get the image center on the target coordinate (radius = half the square size) """
        instru = self.get_instrument(band, **kwargs)
        size_pixels = radius * instru.units_to_pixels( runits ).value
        xpix, ypix = instru.coords_to_pixel(instru.target.ra, instru.target.dec)
        ymin,ymax,xmin,xmax = int(np.rint(ypix-size_pixels)),int(np.rint(ypix+size_pixels)), int(np.rint(xpix-size_pixels)),int(np.rint(xpix+size_pixels))
        return (xpix-xmin,ypix-ymin), instru.data[ymin:ymax,xmin:xmax]
        


    def show_stamps(self, savefile=None, figsize=None, colorbase="white", radius_kpc=10,
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


        fig = mpl.figure( figsize=figsize )
        width, spanx = 0.8/ninstruments, 0.02
        fullwidth = 0.8 + spanx*(ninstruments-1)
        edges = 1-fullwidth
        ax = {}
        for i,band  in enumerate(instrumentnames):
            ax[band] = fig.add_axes([edges/2+i*(width+spanx),0.15,width,0.75])

        prop = {**dict( origin="lower"), **kwargs}

        for band in self.instruments.keys():
            if radius_kpc is not None:
                centroid, data = self.get_localdata(band, radius_kpc, "kpc")
            else:
                instru = self.get_instrument(band)
                data  = instru.data
                centroid = instru.coords_to_pixel(*self.target.radec)
                
            if stretching is not None:
                data = getattr(np,stretching)(data)
            vmin_, vmax_ = parse_vmin_vmax(data, vmin, vmax)
            ax[band].imshow( data, vmin=vmin_, vmax=vmax_, cmap=CMAPS[band], **prop)

            arcsec_kpc = self.get_instrument(band).units_to_pixels( "kpc" ).value
            ax[band].add_patch( 
                    Circle(centroid, radius = 1*arcsec_kpc,
                      facecolor="None", edgecolor=linecolor, lw=1.5)
                    )
            ax[band].add_patch( 
                    Circle(centroid, radius = 3*arcsec_kpc,
                      facecolor="None", edgecolor=linecolor, lw=0.5, ls="--")
                    )
            ax[band].text(0.02,0.98, f"{band}", va="top",ha="left", 
                          transform=ax[band].transAxes, color=textcolor, fontsize="small")

        [ax.set_yticks([]) for ax in fig.axes]
        [ax.set_xticks([]) for ax in fig.axes]

        fig.axes[0].set_ylabel(self.target.name)

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

    BANDS = {"ps1.g":{"lbda":4866.457871},
             "ps1.r":{"lbda":6214.623038},
             "ps1.i":{"lbda":7544.570357},
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
            ppc = g_.get_target_photopoint(radius, runits=runits, on="data")
            self._photopoints[bandname] = ppc
            gout = ppc.data.copy()
            gout["flux"] = ppc.flux
            gout["flux.err"] = np.sqrt(ppc.var)
            gout["mag"] = ppc.mag
            gout["mag.errasym"] = ppc.mag_err
            gout["mag.err"] = np.mean(ppc.mag_err)
            gout["target.x"],gout["target.y"] = g_.coords_to_pixel(*self.target.radec)
 
            dataout[ppc.bandname] = gout
            
        self._data = pandas.DataFrame(dataout).T
        return self.data
    
    # ================= #
    #  Special Methods  #
    # ================= #
    def get_mass(self, radius=None, runits="kpc", refsize=None):
        """ """
        if radius is not None:
            self.measure_photometry(radius=radius,runits=runits)
            
        from astrobject.collections import photodiagnostics
        self.mass_estimator = photodiagnostics.get_massestimator([self.photopoints["g"],self.photopoints["i"]])
        self.mass_estimator.draw_samplers(distmpc=self.target.distance.to("Mpc").value)
        mass = self.mass_estimator.get_estimate()
        if refsize is not None:
            ref_surface = np.pi*refsize**2
            mass[0] -= np.log10(self.surface.value) - np.log10(ref_surface)
        return mass
        
    def get_backup_mass(self):
        """ Assuming the mean and the std of the local mass distribution from Rigault et al. 2018"""
        return 8.06, 0.58 #from Rigault et al. 2018 (SNfactory data)
        
#               #
#   GALEX       #
#               #
class UVLocalSFR( _Photomatize_ ):
    """ """
    BANDS = {"nuv":{"lbda":2315.66},
             "fuv":{"lbda":1538.62},
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
    
    # ================= #
    #  Special Methods  #
    # ================= #
    def get_sfr(self, lum_fuv_hz=None, coef=1.08, apply_dustcorr=True, inlog=True,
                    radius=None, runits="kpc", **kwargs):
        """ 
        coef = 1.08 if Salim 2007 1.4 for Kennicutt 1998
        
        lum_fuv_hz if provided, in erg/s/cm2/Hz-1

        Could also be implemented:
        https://www.aanda.org/articles/aa/pdf/2015/12/aa26023-15.pdf
        -> SFR (M yr−1) = 4.6 × 10−44 × L(FUVcorr), (here L in erg/s/A)
        """
        if lum_fuv_hz is None:
            lum_fuv_hz = np.asarray( self.get_lfuv(apply_dustcorr=apply_dustcorr, inhz=True,
                                                       radius=radius, runits=runits, **kwargs)
                                         )
            
        sfr, sfr_err = coef*1e-28*lum_fuv_hz
        if not inlog:
            return sfr, sfr_err
        
        return np.log10(sfr),  1/np.log(10) * sfr_err/sfr
    
    def get_lfuv(self, apply_dustcorr=True, surface_density=True, inhz=False,
                     radius=None, runits="kpc"):
        """ get the fuv luminosity 
        
        Parameters
        ----------
        apply_dustcorr: [bool] -optional-
            Shall the magnitude be corrected for host interstellar dust.
            This is based on the uv color (see self.get_afuv()).
            Careful: This should only be applied if you think there indeed is dust.

        surface_density: [bool] -optional-
            Shall the returned luminosity be a surface brightness lumonisity.
            (see f/(4*pi*r^2) )

        inhz: [bool] -optional-
            unit in hz (not AA)

        Returns
        -------
        float, float (value, error)
        """
        if radius is not None:
            self.measure_photometry(radius, runits=runits)
            
        print(" = missing MW CORR =")
        
        f_fuv, f_fuv_err = self.get_photometry("fuv", mag=False)
        _signal_to_noise = f_fuv_err/f_fuv
        
        if apply_dustcorr:
            f_fuv /= 10**(-0.4*self.get_afuv()[0])
            
        if surface_density:
            f_fuv /= self.surface.value
            
        lum_fuv = f_fuv*(4*np.pi*self.target.distance.to("cm").value**2)
        
        if inhz:
            lum_fuv *= (self.BANDS["fuv"]["lbda"]**2/constants.c.to("AA/s")).value
                
        return lum_fuv, lum_fuv*_signal_to_noise
        
    def get_afuv(self, from_prior=True, **kwargs):
        """ Measure the expected FUV absoption assuming Salim et al. 2007 relation based on fuv-nuv color. """
        if not from_prior:
            return salim_afuv(self.get_uvcolor()[0],**kwargs), None
        
        return self.uvprior.get_afuv(*self.get_uvcolor())

    def get_uvcolor(self):
        """ """
        nuv = self.get_photometry("nuv", mag=True)
        fuv = self.get_photometry("fuv", mag=True)
        return fuv[0]-nuv[0],np.sqrt(fuv[1]**2+nuv[1]**2)
    #
    # // Backup NUV
    #
    def get_nuvbackup_sfr(self, radius=None, runits="kpc",
                              uvcolor=None, afuv=None, fullbackup=False,
                              inlog=True, surface_density=True, **kwargs):
        """ 
        uvcolor = fuv-nuv => fuv = uvcolor+nuv
        """
        if radius is not None:
            self.measure_photometry(radius, runits=runits)

        print("missing MW CORR")
        if uvcolor is None or afuv is None:
            uvcolor_, afuv_ = self.draw_uvcolor_afuv(1000)
        if uvcolor is None:
            uvcolor = uvcolor_
        if afuv is None:
            afuv = afuv_
        
        backup = {"uvcolor":uvcolor, "afuv":afuv}
        sfr, sfrerr = self._measure_single_nuvbackup_sfr_(**{**backup,**kwargs})
        if fullbackup:
            return sfr, sfrerr
        low_, med_, up_ = np.percentile(sfr, [16,50,84])
        return med_, np.mean([med_-low_,up_-med_ ])

    def _measure_single_nuvbackup_sfr_(self, uvcolor, afuv, fullbackup=False,
                              inlog=True, surface_density=True, **kwargs):
        """ """
        mag_nuv,mag_err = self.get_photometry("nuv", mag=True)
        mag_fuv = mag_nuv+uvcolor
        flux_fuv_hz = 10**(-0.4*(mag_fuv - afuv + 48.6))
        if surface_density:
            flux_fuv_hz /= self.surface.value

        lum_fuv_hz = flux_fuv_hz*(4*np.pi*self.target.distance.to("cm").value**2)
        return self.get_sfr(lum_fuv_hz=np.asarray([lum_fuv_hz,np.NaN]), inlog=inlog, **kwargs)

    def draw_uvcolor_afuv(self, size=100, **kwargs):
        """ """
        return self.uvprior.draw_prior(size,**kwargs)
        
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def uvprior(self):
        """ """
        if not hasattr(self,"_uvprior"):
            self._uvprior = UVPrior()
        return self._uvprior
    

class UVPrior():
    """ """
    _DEFAULT_PARAM = dict(mean_uvcol=0.5, mean_afuv=1.8, sigma_uvcol=0.15, sigma_afuv=0.55, rho=0.87)

    def draw_prior(self, size, **kwargs):
        """ """
        mean_uvcol, mean_afuv, sigma_uvcol, sigma_afuv, rho = self._read_kwargs_()
        return stats.multivariate_normal(mean=[mean_uvcol,mean_afuv],
                                        cov=[[sigma_uvcol**2,rho*sigma_afuv*sigma_uvcol],
                                        [rho*sigma_afuv*sigma_uvcol, sigma_afuv**2]]).rvs(size).T
    
    def _read_kwargs_(self, **kwargs):
        """ """
        prop = {**self._DEFAULT_PARAM,**kwargs}
        return [prop[k] for k in ["mean_uvcol","mean_afuv","sigma_uvcol","sigma_afuv","rho"]]

    def get_afuv(self, uvcolor, uvcolor_err, **kwargs):
        """ """
        posterior = self.draw_posterior(uvcolor, uvcolor_err, **kwargs)
        return np.mean(posterior[1]), np.std(posterior[1])

    def draw_posterior(self, uvcolor, uvcolor_err, size=1000, which="afuv", **kwargs):
        """ """
        ndraw_prior= size*10
        prior_uvcolor_afuv = self.draw_prior(ndraw_prior, **kwargs)
        
        post_uvcolor = stats.norm.pdf(prior_uvcolor_afuv[0],
                                            loc=uvcolor, scale=uvcolor_err)
        rand_index = np.random.choice(np.arange(ndraw_prior), size=size,
                                          p=post_uvcolor/post_uvcolor.sum())
        return prior_uvcolor_afuv.T[rand_index].T
        

    def show(self, data=None):
        """ """
        import matplotlib.pyplot as mpl
        fig = mpl.figure()
        ax = fig.add_subplot(111)

        ax.scatter(*self.draw_prior(10000), facecolors="0.7", edgecolors="None", s=10, alpha=0.1)

        if data is not None:
            data_,error_ = data
            ax.axvline(data_, color="C0", ls="--")
            ax.axvspan(data_-error_, data_+error_, color="C0", alpha=0.3)
            ax.scatter(*self.draw_posterior(*data,100), facecolors="C0", edgecolors="None", marker="x", s=10, alpha=0.1)
            afuv, afuverr = self.get_afuv(*data)
            ax.errorbar(data_, afuv, yerr=afuverr, marker="s", ms=10, color="C0", ecolor="C0")
        
