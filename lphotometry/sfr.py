from astropy import  constants

from astrobject.utils import tools
from scipy import stats
import numpy as np



def salim_afuv(uvcolor, use_sfcurve=True, squeeze=True):
    """ uvcolor = FUV-NUV restframe see Salim et al. 2007 """
    coef, offset,cut, backup = (3.32, 0.22, 0.95, 3.37) if use_sfcurve else \
                               (2.99, 0.27, 0.90, 2.96)
                               
    uvc = np.atleast_1d(uvcolor)
    afuv_base = coef*uvc + offset
    afuv_base[uvc>cut] = backup
    
    return np.squeeze(afuv_base) if squeeze else afuv_base



class GalexSFREstimator():

    BANDS = {"nuv":{"lbda":2315.66},
             "fuv":{"lbda":1538.62},
            }

    def __init__(self, fuv=None, nuv=None, distmpc=None):
        """ gmag and imag must be 2d values (value, error)"""
        self.set_magnitudes(fuv=fuv, nuv=nuv)
        self.set_distmpc(distmpc)

    # -------- #
    #   I/O    #
    # -------- #
    
    @classmethod
    def from_magnitudes(cls, fuv, fuv_err, nuv, nuv_err, distmpc):
        """ """
        return cls(fuv=[fuv, fuv_err], nuv=[nuv, nuv_err], distmpc=distmpc)

    @classmethod
    def mag_to_sfr(cls, fuv, fuv_err, nuv, nuv_err, distmpc,
                        apply_dustcorr=True, **kwargs):
        """ """
        this = cls.from_magnitudes(gmag, gmag_err, imag, imag_err, distmpc)
        return this.get_sfr(apply_dustcorr=apply_dustcorr, **kwargs)
    # -------- #
    #  SETTER  #
    # -------- #
    def set_magnitudes(self, fuv, nuv):
        """ gmag and imag must be 2d values (value, error)"""
        self._fuv = fuv
        self._nuv = nuv
        
    def set_distmpc(self, distmpc):
        """ """
        self._distmpc = distmpc

    # -------- #
    #  GETTER  #
    # -------- #
    def get_photometry(self, which, mag=False):
        """ """
        which = which.lower()
        if which == "nuv":
            mag_, emag_ = self.nuv
        elif which == "fuv":
            mag_, emag_ = self.fuv
        else:
            raise ValueError("only nuv and fuv are accepted")

        if mag or mag_ is None:
            return mag_, emag_
        
        return tools.mag_to_flux(mag_, emag_, wavelength=self.BANDS[which]["lbda"])
        
    def get_sfr(self, lum_fuv_hz=None, coef=1.08, apply_dustcorr=True, inlog=True,
                    surface=None, accept_backup=True, **kwargs):
        """ 
        coef = 1.08 if Salim 2007 1.4 for Kennicutt 1998
        
        lum_fuv_hz if provided, in erg/s/cm2/Hz-1

        Could also be implemented:
        https://www.aanda.org/articles/aa/pdf/2015/12/aa26023-15.pdf
        -> SFR (M yr−1) = 4.6 × 10−44 × L(FUVcorr), (here L in erg/s/A)
        """
        if lum_fuv_hz is None:
            lum_fuv_hz = np.asarray( self.get_lfuv(apply_dustcorr=apply_dustcorr,
                                                       inhz=True,accept_backup=accept_backup,
                                                       surface=surface, **kwargs)
                                    )
            
        sfr, sfr_err = coef*1e-28*lum_fuv_hz
        if not inlog:
            return sfr, sfr_err
        
        return np.log10(sfr),  1/np.log(10) * sfr_err/sfr
        
    def get_afuv(self, from_prior=True, **kwargs):
        """ Measure the expected FUV absoption assuming Salim et al. 2007 relation based on fuv-nuv color. """
        if not from_prior:
            return salim_afuv(self.get_uvcolor()[0],**kwargs), None
        
        return self.uvprior.get_afuv(*self.get_uvcolor())

    def get_uvcolor(self):
        """ """
        if self.fuv[0] is None or self.nuv[0] is None:
            return np.NaN, np.NaN
        
        return self.fuv[0]-self.nuv[0],np.sqrt(self.fuv[1]**2+self.nuv[1]**2)
    
    def get_lfuv(self, apply_dustcorr=True, surface=None, inhz=False, accept_backup=True):
        """ get the fuv luminosity 
        
        Parameters
        ----------
        apply_dustcorr: [bool] -optional-
            Shall the magnitude be corrected for host interstellar dust.
            This is based on the uv color (see self.get_afuv()).
            Careful: This should only be applied if you think there indeed is dust.

        surface: [bool] -optional-
            Shall the returned luminosity be a surface brightness lumonisity.
            (see f/(4*pi*r^2) )

        inhz: [bool] -optional-
            unit in hz (not AA)

        Returns
        -------
        float, float (value, error)
        """           
        f_fuv, f_fuv_err = self.get_photometry("fuv", mag=False)
        # -> You don't have FUV data        
        if f_fuv is None:
            if not accept_backup:
                warnings.warn("no fuv and accept_backup is False. NaN returned")
                return np.NaN, np.NaN
            
            lum_fuv, elum_fuv = self.get_nuvbackup_lfuv(size=None)
            if surface is not None:
                lum_fuv /= surface
                elum_fuv /= surface
            return lum_fuv, elum_fuv

        # -> You have FUV data
        _signal_to_noise = f_fuv_err/f_fuv
        
        if apply_dustcorr:
            f_fuv /= 10**(-0.4*self.get_afuv()[0])

        lum_fuv = f_fuv*(4*np.pi*self._distcm**2)
        if inhz:
            lum_fuv *= (self.BANDS["fuv"]["lbda"]**2/constants.c.to("AA/s")).value
        
        if surface is not None:
            lum_fuv /= surface

        return lum_fuv, lum_fuv*_signal_to_noise
        

    #
    # // Backup NUV
    #
    def get_nuvbackup_lfuv(self, uvcolor=None, afuv=None, size=None):
        """ """
        if uvcolor is None or afuv is None:
            uvcolor_, afuv_ = self.uvprior.draw_prior(size=1000 if size is None else size)
            
        if uvcolor is None:
            uvcolor = uvcolor_
            
        if afuv is None:
            afuv = afuv_
            
        # - NUV
        mag_nuv,mag_err = self.get_photometry("nuv", mag=True)
        # -> FUV
        mag_fuv = mag_nuv+uvcolor
        # -> flux_FUV
        flux_fuv_hz = 10**(-0.4*(mag_fuv - afuv + 48.6))
        # -> lum_fuv
        f_ = flux_fuv_hz*(4*np.pi*self._distcm**2)

        if size is None:
            return np.mean(f_), np.std(f_)
        return f_


    # -------- #
    #  PLOTS   #
    # -------- #        
    def show_sfr(self, ax=None, savefile=None, surface=None,
                     clear_axes=["left","right","top"],
                    color="purple", color_nodust=None, set_label=True, r13_color="k"):
        """ 
        if color_nodust is None: color_nodust=color with alpha 0.2
        
        r13_color is None, no line
        """
        import matplotlib.pyplot as mpl
        from matplotlib.colors import to_rgba
        if ax is None:
            fig = mpl.figure(figsize=[6,1.5])
            ax = fig.add_axes([0.1, 0.4,0.8,0.5])
        else:
            fig = ax.figure

        if color_nodust is None:
            color_nodust = to_rgba(color, 0.2)
                
        sfr_dustcorr = self.get_sfr(apply_dustcorr=True, surface=surface)
        sfr_nodustcorr = self.get_sfr(apply_dustcorr=False, surface=surface)

        
        yy = np.linspace(sfr_nodustcorr[0]-4*sfr_nodustcorr[1],
                             sfr_nodustcorr[0]+4*sfr_nodustcorr[1], 100)
        ax.fill_between(yy, stats.norm.pdf(yy, *sfr_nodustcorr), 
                            facecolor=color_nodust, label="no dust correction")

        yy = np.linspace(sfr_dustcorr[0]-4*sfr_dustcorr[1],
                             sfr_dustcorr[0]+4*sfr_dustcorr[1], 100)

        ax.plot(yy, stats.norm.pdf(yy, *sfr_dustcorr), color=color_nodust, lw=1.5)
        if r13_color is not None:
            ax.axvline(-2.9, lw=1, color=r13_color, ls="--")
            
        ax.set_ylim(0)

        if set_label:
            ax.set_xlabel(r"$\log(\Sigma_{\mathrm{SFR}}\ [\mathrm{M_\odot\, yr^{-1}\, kpc^{-2}}])$", fontsize="large")

        if clear_axes is not None:
            [ax.spines[which].set_visible(False) for which in clear_axes]
            
        ax.set_yticks([])

        if savefile is not None:
            fig.savefig(savefile)
            
        return fig
        
    def show_afuv(self, ax=None, savefile=None,
                      color_prior="0.2", color_data="C0",
                      color_posterior="C1", color_inferred=None,
                      set_legend=True, set_label=True, ncol_legend=4):
        """ 
        color_inferred: [string/None] 
            if None, same as color_posterior
        """
        import matplotlib.pyplot as mpl
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Ellipse


        from . import tools
        
        if ax is None:
            fig = mpl.figure(figsize=[6,4])
            ax  = fig.add_axes([0.15,0.15,0.75,0.75])
        else:
            fig = ax.figure
        
        if color_inferred is None:
            color_inferred = color_posterior
            
        # - Data
        color, dcolor = self.get_uvcolor()
        ax.axvspan(color-2*dcolor, color+2*dcolor, color=to_rgba(color_data, 0.1),
                  label="Observed")
        ax.axvspan(color-3*dcolor, color+3*dcolor, color=to_rgba(color_data, 0.1))

        # - Prior        
        ellipse = Ellipse(**self.uvprior.get_std_ellipse(2), 
                          facecolor=to_rgba(color_prior, 0.3), edgecolor=to_rgba(color_prior, 1), lw=0.5,
                          label="SF Galaxies")
        ax.add_patch(ellipse)

        ellipse = Ellipse(**self.uvprior.get_std_ellipse(3), 
                          facecolor=to_rgba(color_prior, 0.1), edgecolor=to_rgba(color_prior, 0.5), lw=0.5
                          )
        ax.add_patch(ellipse)


        # - Inferred
        colors = self.get_uvcolor()
        if not np.isnan(colors[0]):
            x,y = self.uvprior.draw_posterior(*colors, size=10000, prior_times=100)
        else:
            x,y = self.uvprior.draw_prior(size=10000)
        # -- Posterior
        _ = tools.confidence_ellipse(x,y, ax=ax, n_std=2,
                                   facecolor=to_rgba(color_posterior, 0.5), edgecolor=to_rgba(color_posterior, 1), lw=0.5,
                                   label="Posterior")
        _ = tools.confidence_ellipse(x,y, ax=ax, n_std=3,
                                   facecolor=to_rgba(color_posterior, 0.3), edgecolor=to_rgba(color_posterior, 0.8), lw=0.5
                                    )
            # -- Posterior on A_fuv
        yy = np.linspace(-1,5, 1000)
        pdf_ = stats.norm.pdf(yy, loc=np.mean(y), scale=np.std(y)) #gaussian_kde(y)(yy)
        ax.plot(pdf_/10, yy, color=color_inferred, lw=1.5,
                    label=r"Inferred $A_{FUV}$")


        ax.set_xlim(0,1.2)
        ax.set_ylim(0,5)

        if set_legend:
            ax.legend(ncol=ncol_legend, loc=[0,1.], frameon=False, fontsize='small')
            
        if set_label:
            ax.set_ylabel(r"$A_{FUV}$", fontsize="large")
            ax.set_xlabel(r"$^0(FUV-NUV)$", fontsize="large")

        if savefile is not None:
            fig.savefig(savefile)
            
        return fig
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def uvprior(self):
        """ """
        if not hasattr(self,"_uvprior"):
            self._uvprior = UVPrior()
        return self._uvprior
        
    @property
    def nuv(self):
        """ """
        return self._nuv
    
    @property
    def _nuvstat(self):
        """ """
        if not hasattr(self,"_hnuvstat"):
            _mag, _emag = self._nuv
            self._hnuvstat = stats.lognorm(s=_emag, loc=_mag-1)
        return self._hnuvstat
        
    @property
    def fuv(self):
        """ """
        return self._fuv
    
    @property
    def _fuvstat(self):
        """ """
        if not hasattr(self,"_hfuvstat"):
            _mag, _emag = self._fuv
            self._hfuvstat = stats.lognorm(s=_emag, loc=_mag-1)
        return self._hfuvstat
        
    @property
    def distmpc(self):
        """ """
        return self._distmpc

    @property
    def _distcm(self):
        """ """
        mpc_to_cm = 3.085677581491367e+24
        return self.distmpc*mpc_to_cm








    
class UVPrior():
    """ """
    _DEFAULT_PARAM = dict(mean_uvcol=0.5, mean_afuv=1.8, sigma_uvcol=0.15, sigma_afuv=0.55, rho=0.87)

    def draw_prior(self, size, **kwargs):
        """ """
        return stats.multivariate_normal(mean=self.mean, cov=self.covariance).rvs(size).T
    
    def _read_kwargs_(self, **kwargs):
        """ """
        prop = {**self._DEFAULT_PARAM,**kwargs}
        return [prop[k] for k in ["mean_uvcol","mean_afuv","sigma_uvcol","sigma_afuv","rho"]]

    def get_afuv(self, uvcolor, uvcolor_err, **kwargs):
        """ """
        posterior = self.draw_posterior(uvcolor, uvcolor_err, **kwargs)
        return np.mean(posterior[1]), np.std(posterior[1])

    def draw_posterior(self, uvcolor, uvcolor_err, size=1000, which="afuv",
                           prior_times=10, **kwargs):
        """ """
        ndraw_prior= size*prior_times
        prior_uvcolor_afuv = self.draw_prior(ndraw_prior, **kwargs)
        
        post_uvcolor = stats.norm.pdf(prior_uvcolor_afuv[0],
                                            loc=uvcolor, scale=uvcolor_err)
        rand_index = np.random.choice(np.arange(ndraw_prior), size=size,
                                          p=post_uvcolor/post_uvcolor.sum())
        return prior_uvcolor_afuv.T[rand_index].T
        
    def show_draws(self, data=None):
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
            

    def get_std_ellipse(self, n_std):
        """ returns the std ellipse parameters as expected by matplotlib.Patches.Ellipse 
        For instance it means that width and height are diagonals.
        (see lphotometry.tools.covariance_to_ellipse )
        """
        from .tools import covariance_to_ellipse
        return covariance_to_ellipse(self.covariance, mean=self.mean, n_std=n_std)

    
    @property
    def mean(self):
        """ """
        return self._DEFAULT_PARAM["mean_uvcol"], self._DEFAULT_PARAM["mean_afuv"]

    @property
    def covariance(self):
        """ """
        sigma_uvcol, sigma_afuv, rho = [self._DEFAULT_PARAM[k] for k in ["sigma_uvcol","sigma_afuv","rho"]]
        return np.asarray([[sigma_uvcol**2,rho*sigma_afuv*sigma_uvcol],
                           [rho*sigma_afuv*sigma_uvcol, sigma_afuv**2]]
                        )
