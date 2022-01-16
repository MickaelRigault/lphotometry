#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from . import io
from pymage import query


def get_instrument(which, target, forcedl=False,
                       **kwargs):
    """ Get an Instrument Object for the given target.

    Parameters
    ----------
    which: [string]
        name of the instrument.
        implemented: galex, panstarrs

    target: [string or Target object]
        if string, the target will be loaded from the given name (assuming it's known)
        if Target this will be used as such.
        
    forcedl: [bool]
        if a file for the target and instrument already exists
        shall this redownload it.

    **kwargs goes to get_{which}()
    """
    if which == "galex":
        return get_galex(target, forcedl=forcedl, **kwargs)
    
    if which == "panstarrs":
        return get_panstarrs(target, forcedl=forcedl, **kwargs)
    
    raise NotImplementedError(f"{which} not implemented")


#               #
#   GALEX       #
#               #
def get_galex(target, forcedl=False, **kwargs):
    """ fetch for the galex instrument associated to the target.
    If this is the first time the target is requested, this will look for the 
    galex data online and will download them (see autodl)

    Parameters
    ----------
    name: [string]
        Target name

    autodl: [bool] -optional-
        If the data do not exist, should this download them ?
        (only works if new target)
    
    Returns
    -------
    list of astrobject's instrument
    """
    if type(target) == str: # it is the target name
        from .target import get_target
        target = get_target(target)

    name = target.name
    qgalex = query.GALEXQuery()
    if not qgalex.is_target_known(name):
        qgalex.download_target_metadata(name, *target.radec)
        qgalex.store()

    # download if needed only
    _ = qgalex.download_target_data(name, overwrite=forcedl, verbose=False)
    galexd = qgalex.get_target_instruments(name, **kwargs)
    galexout = {}
    for galex_ in galexd:
        bandname = galex_.bandname
        if bandname in galexout.keys():
            k_=0
            while bandname+"_%d"%k_ in galexout.keys():
                k_+=1
            label = bandname+"_%d"%k_
        else:
            label = bandname
        galexout[label] =galex_

    return galexout

def download_panstarrsinst(target, filters = ["g","r","i"], buffersize=10,
                               pixel_in_arcsec=0.25, run_sep=False):
    """ """
    from pymage.panstarrs import PS1Target
    if type(target) == str: # it is the target name
        from .target import get_target
        target = get_target(target)
        
    
    ps = PS1Target.from_coord(*target.radec)
    ps.download_cutout(size=int(buffersize*target.arcsec_per_kpc.value/pixel_in_arcsec), filters=filters, run_sep=run_sep)
    return ps.imgcutout

def get_panstarrs(target, bands=["g","r","i"], forcedl=False, storenew=True, radius_kpc=50,
                      background=0, **kwargs):
    """ fetch for the galex instrument associated to the target.
    If this is the first time the target is requested, this will look for the 
    galex data online and will download them (see autodl)

    Parameters
    ----------
    target: [string]
        Target name or Target object

    autodl: [bool] -optional-
        If the data do not exist, should this download them ?
        (only works if new target)
    
    Returns
    -------
    list of astrobject's instrument
    """
    
    name = target if type(target) == str else target.name
        
    from astrobject.instruments import panstarrs
    filenames = io.get_panstarrs_files(name, bands=bands)
    if forcedl:
        missing_bands = bands
    else:
        missing_bands = [k for k, v in filenames.items() if v is None]
        
    if len(missing_bands)>0:
        missing_inst = download_panstarrsinst(target, filters=missing_bands, buffersize=radius_kpc, **kwargs)
        if storenew:
            for band_,inst_ in missing_inst.items():
                fileout = io._panstarrs_filename_(name, band_)
                os.makedirs(os.path.dirname(fileout), exist_ok=True)
                inst_.writeto(fileout)
    else:
        missing_inst = {}

    stored_inst = {band_: panstarrs.PanSTARRS(file_, background=background) for band_, file_ in filenames.items() if file_ is not None}
    return {**missing_inst,**stored_inst}
 
