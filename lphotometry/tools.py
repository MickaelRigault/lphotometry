#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.cosmology import Planck15 as cosmo

def find_zcosmo(distmod, zguess=0.001):
    from scipy.optimize import fmin
    def _to_minimize_(z):
        if z<0:
            return 1000
        return np.abs(cosmo.distmod(z).value-distmod)
    
    return fmin(_to_minimize_, zguess, disp=0)


def rebin_arr(arr, bins, use_dask=False):
    ccd_bins = arr.ravel().reshape( int(arr.shape[0]/bins[0]), 
                                    bins[0],
                                    int(arr.shape[1]/bins[1]), 
                                    bins[1])
    if use_dask:
        return da.moveaxis(ccd_bins, 1,2)
    return np.moveaxis(ccd_bins, 1,2)

def parse_vmin_vmax(data, vmin, vmax):
    """ Parse the input vmin vmax given the data.\n    
    If float or int given, this does nothing \n
    If string given, this computes the corresponding percentile of the data.\n
    e.g. parse_vmin_vmax(data, 40, '90')\n
    -> the input vmin is not a string, so it is returned as such\n
    -> the input vmax is a string, so the returned vmax corresponds to the 
       90-th percent value of data.

    Parameters
    ----------
    data: array     
        data (float array)

    vmin,vmax: string or float/int
        If string, the corresponding percentile is computed\n
        Otherwise, nothing happends.

    Returns
    -------
    float, float
    """
    if vmax is None: vmax="99"
    if vmin is None: vmin = "1"
                
    if type(vmax) == str:
        vmax=np.nanpercentile(data, float(vmax))
        
    if type(vmin) == str:
        vmin=np.nanpercentile(data, float(vmin))
        
    return vmin, vmax
