#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" """

import warnings
import pandas
import numpy as np

from . import photometry, io


class UVLocalsSFR( photometry._Photomatize_ ):
    """ """
    def load_data(self, **kwargs):
        """ """
        self.uv = photometry.UVLocalSFR.from_target(self.target, **kwargs)
        self.optical = photometry.PS1LocalMass.from_target(self.target, **kwargs)
        
    def set_instrument(self, *args, **kwargs):
        """ """
        raise NotImplementedError("You cannot set instrument directly from UVLocalsSFR")

    def measure_photometry(self, radius, runits="kpc"):
        """ """
        _ = self._local_photometry_setup_(radius, runits=runits)
        self.uv.measure_photometry(radius, runits=runits)
        self.optical.measure_photometry(radius, runits=runits)
        # Set the data of this object
        self._data = pandas.concat([self.uv.data, self.optical.data], sort=False)
        self._derive_parameters_()

    # ================= #
    #  Special Methods  #
    # ================= #
    def get_lssfr(self, radius=None, runits="kpc", allow_backup=["mass","sfr"], **kwargs):
        """ """
        if allow_backup is None:
            allow_backup = []

        if radius is not None:
            self.measure_photometry(radius, runits=runits)
            
        local_mass = self.get_mass(allow_backup = "mass" in allow_backup)
        local_sfr  = self.get_sfr(allow_backup = "sfr" in allow_backup)
        return local_sfr[0] - local_mass[0], np.sqrt(local_sfr[1]**2 + np.mean(local_mass[1:])**2)
        
    def get_mass(self, radius=None, runits="kpc", allow_backup=True, **kwargs):
        """ """
        if radius is not None:
            self.measure_photometry(radius, runits=runits)

            
        if self.has_optical():
            m, *err = self.optical.get_mass(**kwargs)
            return m, np.mean([err])
        
        elif allow_backup:
            return self.optical.get_backup_mass(**kwargs)
        
        return np.NaN,np.NaN

    def get_sfr(self, radius=None, runits="kpc", allow_backup=True, **kwargs):
        """ """
        if not self.has_uv():
            return np.NaN,np.NaN

        if radius is not None:
            self.measure_photometry(radius, runits=runits)
        
        if self.has_fuv():
            return self.uv.get_sfr(**kwargs)
        elif allow_backup:
            return self.uv.get_nuvbackup_sfr(**kwargs)
        else:
            return np.NaN,np.NaN
            
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
            
    def get_derived_parameters(self, rebuild=False, radius=None, runits="kpc"):
        """ """
        if radius is not None:
            self.measure_photometry(radius, runits=runits) # this inclide a get_derived_parameters already
            return self._derived_data
            
        if not rebuild and hasattr(self,"_derived_data") and self._derived_data is not None:
            return self._derived_data
        
        data = {}
        # = Photometry
        data["nuv"],data["nuv.err"] = self.get_photometry("nuv")
        if self.has_fuv():
            data["fuv"],data["fuv.err"] = self.get_photometry("fuv")
        else:
            data["fuv"],data["fuv.err"] = None,None

        if self.has_optical():
            for k in ["g","r","i"]:
                data[k],data[k+".err"] = self.get_photometry(f"ps1.{k}")
        else:
            for k in ["g","r","i"]:
                data[k],data[k+".err"] = None,None
                
        # = Generic
        data["distmpc"] = self.target.distance.to("Mpc").value
        data["rad_arcsec"] = self.arcsec_per_kpc
        data["name"] = self.target.name
        
        # = SFR
        data["log_sfr"], data["log_sfr.err"] = self.get_sfr(inlog=True)
        data["log_sfr.isbackup"] = not self.has_fuv()
        
        # Mass
        data["lmass"], data["lmass.err"] = self.get_mass()
        data["lmass.isbackup"] = not self.has_optical()
        # LsSFR
        data["lssfr"], data["lssfr.err"] = self.get_lssfr()
        # NUV-r
        data["nuvr"],data["nuvr.err"] = self.get_nuvr()

        return pandas.Series(data)

    def _derive_parameters_(self):
        """ """
        self._derived_data = self.get_derived_parameters(rebuild=True)
        
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
        return len(self.optical.bandnames)>0
    
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
