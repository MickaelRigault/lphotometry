import numpy as np
from scipy import stats

class NUVRPrior():
    BR_RATIO = 1/1.6
    BG_RATIO = 1/7
    _DEF_RANGE = [-1, 10]
    _PDF_AMPLE = 1/(1+ BR_RATIO + BG_RATIO)
    
    def get_pdf(self, x):
        """ """
        return self._PDF_AMPLE * (self._blues.pdf(x) + \
                                    self._reds.pdf(x)*self.BR_RATIO + \
                                    self._greens.pdf(x)*self.BG_RATIO)
    
    def show(self, ax = None, show_details=True, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        
        if ax is None:
            fig = mpl.figure(figsize=[6,4])
            ax  = fig.add_axes([0.15,0.15,0.75,0.75])
        else:
            fig = ax.figure
        
        xx = np.linspace(*self._DEF_RANGE, 1000)
        ax.plot(xx, self.get_pdf(xx), color="0.2", label="Galaxies")
        ax.set_ylim(0)

        fill_prop = {"alpha":0.05, "edgecolor":"None"}
        ax.fill_between(xx, self._blues.pdf(xx)*self._PDF_AMPLE, 
                        color="slategrey", **fill_prop)
        base = self._blues.pdf(xx)*self._PDF_AMPLE


        ax.fill_between(xx, self._reds.pdf(xx)*self._PDF_AMPLE*self.BR_RATIO,
                        color="tab:red", **fill_prop)
        base += self._reds.pdf(xx)*self._PDF_AMPLE*self.BR_RATIO


        ax.fill_between(xx, self._greens.pdf(xx)*self._PDF_AMPLE*self.BG_RATIO+base,
                        y2=base,
                        color="tab:green", **fill_prop)        
        
        return fig
    
    # =============== #
    #   Properties    #
    # =============== #
    @property
    def _blues(self):
        """ """
        return stats.skewnorm(loc=2., scale=0.9, a=1.7)

    @property
    def _reds(self):
        """ """
        return stats.skewnorm(loc=6.1, scale=1.1, a=-4)
    
    @property
    def _greens(self):
        """ """
        return stats.norm(loc=3.8, scale=.5)
