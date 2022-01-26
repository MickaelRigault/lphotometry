#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" """

import warnings
import pandas
import numpy as np
from scipy import stats

from . import photometry, io


class UVLocalsSFR( photometry._Photomatize_ ):
    """ """
    BANDS = {**photometry.UVLocalSFR.BANDS,
             **photometry.PS1LocalMass.BANDS}
    
    def load_data(self, galex_edgebuffer=True, **kwargs):
        """ """
        if not galex_edgebuffer:
            propuv = dict(buffer_safe_width=0)
        else:
            propuv = {}
            
        self.uv = photometry.UVLocalSFR.from_target(self.target, **{**propuv,**kwargs})
        self.optical = photometry.PS1LocalMass.from_target(self.target, **kwargs)
        
    def set_instrument(self, *args, **kwargs):
        """ """
        raise NotImplementedError("You cannot set instrument directly from UVLocalsSFR")

    def measure_photometry(self, radius, runits="kpc", derive_data=True):
        """ """
        _ = self._local_photometry_setup_(radius, runits=runits)
        self.uv.measure_photometry(radius, runits=runits)
        self.optical.measure_photometry(radius, runits=runits)
        # Set the data of this object
        self._data = pandas.concat([self.uv.data, self.optical.data], sort=False)
        if derive_data:
            self._derive_parameters_()

    # ================= #
    #  Special Methods  #
    # ================= #
    def get_lssfr(self, radius=None, runits="kpc",
                      accept_backup=["mass","sfr"], refsize=1,
                      apply_dustcorr=True, **kwargs):
        """ """
        if accept_backup is None:
            accept_backup = []

        if radius is not None:
            self.measure_photometry(radius, runits=runits)
            
        local_mass = self.get_mass(refsize=refsize, accept_backup = "mass" in accept_backup)
        local_sfr  = self.get_sfr(apply_dustcorr=apply_dustcorr,
                                      accept_backup = "sfr" in accept_backup)
    
        return local_sfr[0] - local_mass[0], np.sqrt(local_sfr[1]**2 + np.mean(local_mass[1:])**2)
        
    def get_mass(self, radius=None, runits="kpc", accept_backup=True, refsize=1, **kwargs):
        """ """
        if radius is not None:
            self.measure_photometry(radius, runits=runits)

        return self.optical.get_mass(refsize=refsize, accept_backup=accept_backup,
                                        **kwargs)
        

    def get_sfr(self, radius=None, runits="kpc", accept_backup=True,
                    apply_dustcorr=True, **kwargs):
        """ """
        if not self.has_uv():
            return np.NaN,np.NaN

        if radius is not None:
            self.measure_photometry(radius, runits=runits)
        
        return self.uv.get_sfr(apply_dustcorr=apply_dustcorr,
                                   accept_backup=accept_backup,  **kwargs)
            
    def get_nuvr(self,radius=None, runits="kpc"):
        """ """
        if radius is not None:
            self.measure_photometry(radius, runits=runits)

        warnings.warn("NUV-r not corrected for MW DUST")
        nuv, nuverr = self.uv.get_photometry("nuv", mag=True)
        if self.has_optical():
            r, rerr = self.optical.get_photometry("ps1.r", mag=True)
        else:
            r, rerr = np.NaN, np.NaN
            
        return nuv-r, np.sqrt(nuverr**2+rerr**2)
            
    def get_derived_parameters(self, radius=None, runits="kpc", rebuild=False, refsize=1):
        """ """
        if radius is not None:
            self.measure_photometry(radius, runits=runits, derive_data=False)
            rebuild = True
            
        if not rebuild and hasattr(self,"_derived_data") and self._derived_data is not None:
            return self._derived_data
        
        data = {}
        # = Photometry
        if self.has_uv():
            data["nuv"],data["nuv.err"] = self.get_photometry("nuv")
        else:
            data["nuv"],data["nuv.err"] = None, None
            
        if self.has_fuv():
            data["fuv"],data["fuv.err"] = self.get_photometry("fuv")
        else:
            data["fuv"],data["fuv.err"] = None, None

        if self.has_optical():
            for k in ["g","r","i"]:
                data[k],data[k+".err"] = self.get_photometry(f"ps1.{k}")
        else:
            for k in ["g","r","i"]:
                data[k],data[k+".err"] = None, None
                
        # = Generic
        data["distmpc"] = self.target.distance.to("Mpc").value
        data["rad_arcsec"] = self.arcsec_per_kpc
        data["name"] = self.target.name
        
        # = SFR
        if self.has_uv():
            data["log_sfr"], data["log_sfr.err"] = self.get_sfr(inlog=True)
            data["log_sfr.isbackup"] = not self.has_fuv()
        else:
            data["log_sfr"], data["log_sfr.err"] = None, None
            data["log_sfr.isbackup"] = False
        
        # Mass
        data["lmass"], data["lmass.err"] = self.get_mass(refsize=refsize)
        data["lmass.isbackup"] = not self.has_optical()
        # LsSFR
        if self.has_uv():        
            data["lssfr"], data["lssfr.err"] = self.get_lssfr(refsize=refsize)
            # NUV-r
            data["nuvr"],data["nuvr.err"] = self.get_nuvr()
        else:
            data["lssfr"], data["lssfr.err"] = None, None
            # NUV-r
            data["nuvr"],data["nuvr.err"] = None, None

        return pandas.Series(data)

    def _derive_parameters_(self):
        """ """
        self._derived_data = self.get_derived_parameters(rebuild=True)


    def show(self, savefile=None, instrumentnames=["fuv","nuv","g","r","i"],
                 instnamecolor="k", kpclinecolor="k"):
        """ """
        import matplotlib.pyplot as mpl
        fig = mpl.figure(figsize=[10,7])

        if instrumentnames is None:
            instrumentnames = [k for k in self.instruments.keys() if "_" not in k]
            
        ninstruments = len(instrumentnames)

        # - Stamps
        LEFT = 0.05
        RIGHT = 0.0
        TOP = 0.05
        BOTTOM =0.08

        _TOTAL_WIDTH = 1-(LEFT+RIGHT)
        _TOTAL_HEIGTH = 1-(TOP+BOTTOM)

        spanx = 0.02
        width = _TOTAL_WIDTH/ninstruments - spanx 
        fullwidth = _TOTAL_WIDTH + spanx*(ninstruments-1)

        bottom_stamp = 0.72

        axstamps = []
        for i,band  in enumerate(instrumentnames):
            axstamps.append( fig.add_axes([LEFT+i*(width+spanx),bottom_stamp,width,1-TOP-bottom_stamp]) )


        width_colors = 0.275
        bottom_colors = 0.38
        spancolor=0.051
        ax_afuv = fig.add_axes([LEFT, bottom_colors, width_colors, 0.25])
        ax_nuvr = fig.add_axes([LEFT+1*(width_colors+spancolor), bottom_colors, width_colors, 0.25])
        ax_gi   = fig.add_axes([LEFT+2*(width_colors+spancolor), bottom_colors, width_colors, 0.25])

        ax_sfr  = fig.add_axes([LEFT, BOTTOM+0.1+0.05, 0.45-LEFT, 0.06])
        ax_mass = fig.add_axes([LEFT, BOTTOM, 0.45-LEFT,          0.06])
        ax_lssfr = fig.add_axes([0.55, BOTTOM, 0.475-LEFT, 0.075*2+0.05])


        _ = self.show_stamps(ax=axstamps, instrumentnames=instrumentnames,
                                 textcolor=instnamecolor, linecolor=kpclinecolor)
        #
        _ = self.uv.show_afuv(ax=ax_afuv, ncol_legend=2)
        _ = self.show_nuvr(ax=ax_nuvr)
        _ = self.optical.show_gicolor(ax=ax_gi, ncol_legend=2)
        #
        _ = self.uv.show_sfr(ax=ax_sfr, color="tab:blue")
        _ = self.optical.show_mass(ax=ax_mass, color="tab:red")
        _ = self.show_lssfr(ax=ax_lssfr, color="tab:purple")

        [ax_.set_xlabel(ax_.get_xlabel(), fontsize="medium") for ax_ in fig.axes]
        [ax_.set_ylabel(ax_.get_ylabel(), fontsize="medium") for ax_ in fig.axes]

        [ax_.tick_params(labelsize="small") for ax_ in fig.axes]

        if savefile is not None:
            fig.savefig(savefile)
        return fig

        
    def show_nuvr(self, ax=None):
        """ """
        import matplotlib.pyplot as mpl
        from .nuvrcolor import NUVRPrior

        if ax is None:
            fig = mpl.figure(figsize=[6,4])
            ax  = fig.add_axes([0.15,0.15,0.75,0.75])
        else:
            fig = ax.figure

        nuvr_p = NUVRPrior()#.show(ax=ax)
        _ = nuvr_p.show(ax=ax, show_details=True)

        nuvr_,enuvr_  = self.get_nuvr()

        ax.axvspan(nuvr_-3*enuvr_, nuvr_+3*enuvr_, facecolor="C0", alpha=0.1, edgecolor="None", label="Observed")
        ax.axvspan(nuvr_-2*enuvr_, nuvr_+2*enuvr_, facecolor="C0", alpha=0.1, edgecolor="None")
        ax.set_xlabel("NUV-r [mag]", fontsize="large")
        _ = ax.set_yticks([])

        ax.set_xlim(0,9)
        ax.legend(ncol=4, loc=[0,1.], frameon=False, fontsize='small')
        return fig


    def show_lssfr(self, ax=None, savefile=None, offset=None,
                        clear_axes=["left","right","top"],
                        color="purple", color_nodust=None, set_label=True, r13_color="k"):
        """ 
        if color_nodust is None: color_nodust=color with alpha 0.2

        r13_color is None, no line
        """
        import matplotlib.pyplot as mpl
        from matplotlib.colors import to_rgba
        if ax is None:
            fig = mpl.figure(figsize=[6,3])
            ax = fig.add_axes([0.1,0.25,0.8,0.7])
        else:
            fig = ax.figure

        if color_nodust is None:
            color_nodust = to_rgba(color, 0.1)

        _lssfr = self.get_lssfr()
        _lssfr_nodust = self.get_lssfr(apply_dustcorr=False)

        yy = np.linspace(_lssfr_nodust[0]-4*_lssfr_nodust[1],
                         _lssfr_nodust[0]+4*_lssfr_nodust[1], 100)
        ax.fill_between(yy, stats.norm.pdf(yy, *_lssfr_nodust), 
                            facecolor=color_nodust, label="no dust correction")

        yy = np.linspace(_lssfr[0]-4*_lssfr[1],
                         _lssfr[0]+4*_lssfr[1], 100)

        pdf_ = stats.norm.pdf(yy, *_lssfr)
        ax.plot(yy, pdf_, color=color, 
                lw=1.5)

        if r13_color is not None:
            ax.axvline(-10.82, lw=1, color=r13_color, ls="--")
            #top_value = pdf_.max()
            #ax.text(-10.82+0.01, top_value, "prompt", rotation=90, 
            #        va="top", ha="left", fontsize="x-small")
            #ax.text(-10.82-0.01, 0.05*top_value, "delayed", rotation=90, 
            #        va="bottom", ha="right", fontsize="x-small")

        ax.set_ylim(0)

        if set_label:
            ax.set_xlabel(r"$\log(\mathrm{local\, sSFR})$", fontsize="large")

        if clear_axes is not None:
            [ax.spines[which].set_visible(False) for which in clear_axes]

        ax.set_yticks([])



        if savefile is not None:
            fig.savefig(savefile)

        return fig   
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def derived_data(self):
        """ """
        if not hasattr(self,"_derived_data") or self._derived_data is None:
            self._derived_data = self.get_derived_parameters()
        return self._derived_data
    
    def has_uv(self):
        """ """
        return len(self.uv.bandnames)>0
    
    def has_fuv(self):
        """ """
        return "fuv" in self.bandnames

    def has_optical(self):
        """ """
        return self.optical.has_instruments() and len(self.optical.bandnames)>0
    
    @property
    def photopoints(self):
        """ """
        return {**self.uv.photopoints, **self.optical.photopoints}

    @property
    def bandnames(self):
        """ """
        return np.asarray(self.uv.bandnames.tolist()+self.optical.bandnames.tolist())
    
    @property
    def instruments(self):
        """ """
        return {**self.uv.instruments, **self.optical.instruments}

    # ============= #
    #  Properties   #
    # ============= #
    @property
    def arcsec_per_kpc(self):
        """ """
        return self.target.arcsec_per_kpc.value
