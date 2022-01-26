

import pandas
import warnings
from astropy import units
from .io import load_riess_target, load_freedman_target

RIESS_TARGETS = load_riess_target()
FREEDMAN_TARGETS = load_freedman_target()

def get_target(name, **kwargs):
    """ """
    return Target.from_name(name, **kwargs)

#               #
#   TARGET      #
#               #
class Target(): 
    """ """
    def __init__(self, data=None):
        """ """
        if data is not None:
            self.set_data(data)

    # ============= #
    #   I/O         #
    # ============= #            
    @classmethod
    def from_name(cls, name, favor="riess"):
        """ Grabs data from global variable dataframes (RIESS_TARGETS or FREEDMAN_TARGETS)"""
        if favor == "riess" and name in RIESS_TARGETS.index:
            ra, dec, distmof = RIESS_TARGETS.loc[name][["ra","dec","mu"]].values
        else:
            ra, dec, distmof = FREEDMAN_TARGETS.loc[name][["ra","dec","mu_TRGB"]].values
            
        return cls.from_data(name, ra, dec, distmof)

    @classmethod
    def from_data(cls, name=None, ra=None, dec=None, distmod=None, zcmb=None):
        """ """
        if zcmb is None and distmod is not None:
            from .tools import find_zcosmo
            zcmb = find_zcosmo(distmod)[0]
        elif distmod is None and zcmb is not None:
            from .tools import cosmo
            distmod = cosmo.distmod(zcmb).value
        
            
        data = pandas.Series({"name":name, "ra":ra, "dec":dec,
                              "distmod":distmod, "zcmb":zcmb})
        return cls(data)

    # ============= #
    #   Method      #
    # ============= #            
    def set_data(self, dataserie):
        """ set a pandas.Series. Must contain at least name, ra, dec, distmod and/or zcmb  """
        self._data = dataserie
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def data(self):
        """ """
        if not hasattr(self,"_data"):
            return None
        return self._data

    @property
    def name(self):
        """ """
        return self.data["name"]

    @property
    def radec(self):
        """ """
        return self.data[["ra","dec"]].values
    
    @property
    def zcmb(self):
        """ """
        return self.data["zcmb"]
    
    @property
    def distance(self):
        """ distance in pc (astropy Quantity) """
        return 10**((self.data["distmod"]+5)/5)*units.pc

    @property
    def distmpc(self):
        """ distance in Mpc (float) """
        return self.distance.to("Mpc").value
    
    @property
    def arcsec_per_kpc(self):
        """ """
        from .tools import cosmo
        return cosmo.arcsec_per_kpc_proper(self.zcmb)



class _TargetHandler_():
    
    def __init__(self, target=None):
        """ """
        if target is not None:
            self.set_target(target)

    # ============= #
    #  I/O          #
    # ============= #
    @classmethod
    def from_name(cls, name, **kwargs):
        """ """
        return cls( target=Target.from_name(name), **kwargs )
        
    @classmethod
    def from_targetdata(cls,  name=None, ra=None, dec=None, distmod=None, zcmb=None, **kwargs):
        """ **kwargs goes to __init__ others to Target.from_data()"""
        target = Target.from_data(name, ra=ra, dec=dec, distmod=distmod, zcmb=zcmb)
        return cls(target, **kwargs)

    @classmethod
    def from_target(cls, target, **kwargs):
        """ """
        return cls(target, **kwargs)

    # ============= #
    #  Method       #
    # ============= #
    def set_target(self, target):
        """ set to the instance a target object """
        self._target = target
        
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def target(self):
        """ SN Target object """
        if not hasattr(self,"_target"):
            return None
        return self._target
