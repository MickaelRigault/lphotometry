""" Analysis of masses """


from scipy import stats
import numpy as np



def taylor_mass_relation(mag_i, gi_color, distmpc):
    """ Estimate the stellar-mass of a galaxy based on its i and g bands.
    
    This fuction uses the eq. 8 of Taylor et al 2011
    (# The published version of eq. 8 not the Arxiv eq. 8 #)
    http://adsabs.harvard.edu/abs/2011MNRAS.418.1587T

    log M∗/[M⊙] = 1.15 + 0.70(g − i) − 0.4Mi
    
    Error floor of 0.1dex assumed to account for the spread of the
    (g-i) vs. log M∗/[M⊙] relation. See Taylor et al. 2011 (5.3).
    This error is added in quadrature with the errors of g and i.

    Parameters
    ----------
    
    mag_i, gi_color: [(array of) float,(array of) float]
        Restframe magnitude in "i" band and "g-i" color respectively (calibrated for sdss)

    distmpc: [float]
        Distance in Mega parsec of the galaxy

    Return
    ------
    float (the mass of the target galaxy)
    """
    # Rewrite of the Taylor Equation
    # Mi = i - 5*(np.log10(ppoint_i.target.distmpc*1.e6) - 1)
    #      1.15 + 0.70*(g - i) - 0.4*i + (0.4*5*(np.log10(distmpc*1.e6) - 1))
    #      1.15 + 0.70*g - 1.1*i + (0.4*5*(np.log10(distmpc*1.e6) - 1))

    return 1.15 + 0.70*(gi_color) - 0.4*mag_i + (0.4*5*(np.log10(distmpc*1.e6) - 1))



########################
#                      #
#     Priors           #
#                      #
########################

class PriorGIColor():
    _GAUSSIAN_PARAMETERS = dict(mur=1.25, mub=0.85,
                                sigmar=0.1, sigmab=0.3,
                                b_coef=0.9)
    _DEF_RANGE = [-2,4]


    # -------- #
    #  GETTER  #
    # -------- #
    def get_pdf(self, x):
        """ """
        return self._rbratio * self._blues.pdf(x) + (1-self._rbratio)*self._reds.pdf(x)

    # -------- #
    #  DRAWS   #
    # -------- #    
    def draw_prior(self, size):
        """ """
        size = int(size)
        
        blue_samplers = self._blues.rvs(size*2) # random
        red_samplers = self._reds.rvs(size*2) # random
        
        n_blues = int(np.rint(size*self._rbratio))
        n_reds = size-n_blues
        return np.concatenate([blue_samplers[:n_blues],red_samplers[:n_reds]])

    def draw_posterior(self, data, error=None, size=1000, prior_times=10):
        """ """
        ndraw_prior= size*prior_times
        prior = self.draw_prior(ndraw_prior)
        if error is not None:
            posterior = stats.norm.pdf(prior, loc=data, scale=error)
        else:
            posterior = stats.gaussian_kde(data)(prior)
            
        rand_index = np.random.choice(np.arange(ndraw_prior), size=size,
                                          p=posterior/posterior.sum())
        return prior.T[rand_index].T    
    
    # --------- #
    #  PLOTS    #
    # --------- #
    def show(self, data_error=None, ax=None, savefile=None,
                 color_data="C0", color_prior="0.2",
                 color_inferred="C1", posterior_ampl=0.9,
                 ncol_legend=4):
        """ """
        import matplotlib.pyplot as mpl
        from matplotlib.colors import to_rgba
        
        if ax is None:
            fig = mpl.figure(figsize=[6,4])
            ax  = fig.add_axes([0.15,0.15,0.75,0.75])
        else:
            fig = ax.figure
        
        # - prior
        xx = np.linspace(*self._DEF_RANGE, 1000)
        ax.plot(xx, self.get_pdf(xx), color=to_rgba(color_prior), label="Galaxies")
        
        # - data
        if data_error is not None:
            data, error = data_error
            ax.axvspan(data-2*error, data+2*error, color=to_rgba(color_data, 0.1), label="Observed")
            ax.axvspan(data-3*error, data+3*error, color=to_rgba(color_data, 0.1))
        

            posterior = self.draw_posterior(data, error, size=1000)
            gposterior = stats.gaussian_kde(posterior)
            coef = self.get_pdf(np.mean(data))
            xxpost= np.linspace(data-4*error, data+4*error, 1000)
            ax.plot(xxpost, gposterior(xxpost)/gposterior(xxpost).max()*coef*posterior_ampl, 
                    color=color_inferred, lw=1.5,
                   label=r"Inferred")

        ax.set_xlim(-0.5, 2)
        ax.set_ylim(0)
        ax.set_yticks([])
        
        ax.set_xlabel("g-i [mag]", fontsize="large")
        ax.legend(ncol=ncol_legend, loc=[0,1.],
                      frameon=False, fontsize='small')
        if savefile is not None:
            fig.savefig(savefile)
            
        return fig
    
    # ================ #
    #   Properties     #
    # ================ #
    @property
    def _rbratio(self):
        """ """
        return self._GAUSSIAN_PARAMETERS["b_coef"]
    
    @property
    def _blues(self):
        """ """
        if not hasattr(self,"_hblues"):
            self._hblues = stats.norm(loc=self._GAUSSIAN_PARAMETERS["mub"],
                                    scale=self._GAUSSIAN_PARAMETERS["sigmab"])
        return self._hblues
    
    @property
    def _reds(self):
        """ """
        if not hasattr(self,"_hreds"):
            self._hreds = stats.norm(loc=self._GAUSSIAN_PARAMETERS["mur"],
                                    scale=self._GAUSSIAN_PARAMETERS["sigmar"])
        return self._hreds
    
########################
#                      #
#     Mass             #
#                      #
########################

class MassEstimator():
    
    def __init__(self, gmag=None, imag=None, distmpc=None):
        """ gmag and imag must be 2d values (value, error)"""
        self.set_magnitudes(imag=imag, gmag=gmag)
        self.set_distmpc(distmpc)

    # -------- #
    #   I/O    #
    # -------- #    
    @classmethod
    def from_magnitudes(cls, gmag, gmag_err, imag, imag_err, distmpc):
        """ """
        return cls([gmag, gmag_err], [imag, imag_err], distmpc)

    @classmethod
    def mag_to_mass(cls, gmag, gmag_err, imag, imag_err, distmpc,
                        use_giprior=True, **kwargs):
        """ """
        this = cls.from_magnitudes(gmag, gmag_err, imag, imag_err, distmpc)
        return this.get_mass(use_giprior=use_giprior, **kwargs)
    
    # -------- #
    #  SETTER  #
    # -------- #
    def set_magnitudes(self, imag, gmag):
        """ gmag and imag must be 2d values (value, error)"""
        self._imag = imag
        self._gmag = gmag
        
    def set_distmpc(self, distmpc):
        """ """
        self._distmpc = distmpc

    # -------- #
    #  GETTER  #
    # -------- #
    def get_mass(self, use_giprior=True, size=None, offset=None):
        """ 
        offset: [float or None] -optional-
            if you want to retract this to the measured mass.
            This is for instance relevant if you measured the mass in aperture of 3kpc 
            and want the equivalent in 1kpc aperture unit. In that example:
            offset = np.log10(3**2)
            
        """
        if size is None:
            get_mean = True
            size=1000
        else:
            get_mean = False
        
        # remark, not using get_gicolor for imags and gmags to be the same
        gmags, imags = self._gmagstat.rvs(size),self._imagstat.rvs(size)
        colors = gmags-imags
        if use_giprior:
            colors = self.giprior.draw_posterior(colors, size=size)
        
        mass = taylor_mass_relation(imags, colors, self.distmpc)
        if offset is not None:
            mass -= offset
        if get_mean:
            return np.mean(mass), np.std(mass)
        
        return mass
    
    def get_gicolor(self, use_prior=True, size=1000):
        """ if size is None, mean and std returned (based on 1000 samples)  """
        if size is None:
            get_mean = True
            size=1000
        else:
            get_mean = False
            
        # - Colors
        colors = self._gmagstat.rvs(size)-self._imagstat.rvs(size)
        if use_prior:
            colors = self.giprior.draw_posterior(colors)
            
        # - Returns            
        if get_mean:
            return np.mean(colors), np.std(colors)
        
        return colors
    
    # -------- #
    # PLOTTER  #
    # -------- #
    def show_gicolor(self, ax=None, savefile=None,
                      color_prior="0.2", color_data="C0",
                      color_inferred="C1", **kwargs):
        """ """
        return self.giprior.show(self.get_gicolor(use_prior=False, size=None),
                                ax=ax, savefile=savefile,
                                color_prior=color_prior, color_data=color_data,
                                color_inferred=color_inferred,**kwargs)
    
    def show_mass(self, ax=None, savefile=None, offset=None,
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

        mass_prior = self.get_mass(use_giprior=True, size=None, offset=offset)
        mass_noprior = self.get_mass(use_giprior=False, size=None, offset=offset)
        

        yy = np.linspace(mass_noprior[0]-4*mass_noprior[1],
                         mass_noprior[0]+4*mass_noprior[1], 100)
        ax.fill_between(yy, stats.norm.pdf(yy, *mass_noprior), 
                            facecolor=color_nodust, label="no dust correction")

        yy = np.linspace(mass_prior[0]-4*mass_prior[1],
                             mass_prior[0]+4*mass_prior[1], 100)

        ax.plot(yy, stats.norm.pdf(yy, *mass_prior), color=color, 
                lw=1.5)
        if r13_color is not None:
            ax.axvline(8.04, lw=1, color=r13_color, ls="--")

        ax.set_ylim(0)

        if set_label:
            ax.set_xlabel(r"$\log(\mathrm{M_*/M_\odot})$", fontsize="large")

        if clear_axes is not None:
            [ax.spines[which].set_visible(False) for which in clear_axes]

        ax.set_yticks([])

        if savefile is not None:
            fig.savefig(savefile)

        return fig        
        
    # ================ #
    #   Properties     #
    # ================ #  
    @property
    def giprior(self):
        """ """
        if not hasattr(self, "_giprior"):
            self._giprior = PriorGIColor()
        return self._giprior
        
    @property
    def gmag(self):
        """ """
        return self._gmag
    
    @property
    def _gmagstat(self):
        """ """
        if not hasattr(self,"_hgmagstat"):
            _mag, _emag = self._gmag
            self._hgmagstat = stats.lognorm(s=_emag, loc=_mag-1)
        return self._hgmagstat
        
    @property
    def imag(self):
        """ """
        return self._imag
    
    @property
    def _imagstat(self):
        """ """
        if not hasattr(self,"_himagstat"):
            _mag, _emag = self._imag
            self._himagstat = stats.lognorm(s=_emag, loc=_mag-1)
        return self._himagstat
        
    @property
    def distmpc(self):
        """ """
        return self._distmpc
