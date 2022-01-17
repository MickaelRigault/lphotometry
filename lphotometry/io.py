#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings
import pandas
from astropy import units
_SOURCEDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.getenv('DATAPATH',"./Data/")
SPECIAL_DATADIR = DATAPATH+"sn_anchorsample/"

COORDINATES =pandas.DataFrame({'SN1981B':[188.623194, 2.199796],
                               'SN1990N':[190.736174, 13.256495],
                               'SN1994ae':[161.758087, 17.275220],
                               'SN1995al':[147.733208, 33.552611],
                               'SN1998aq':[179.107792, 55.128667],
                               'SN2001el':[56.127500, -44.639833],# inclined !
                               'SN2002fk':[50.523792, -15.400889],
                               'SN2003du':[218.649167, 59.334389],
                               'SN2005cf':[230.384208, -7.413194],
                               'SN2007af':[215.587625, -0.393778],
                               'SN2007sr':[180.469992, -18.972767],
                               'SN2009ig':[39.548375, -1.312528], # NEW
                               'SN2011by':[178.939625, 55.325556], # inclined, NEW !
                               'SN2011fe':[210.774208, 54.273722],
                               'SN2012cg':[186.803458, 9.420333],
                               'SN2012fr':[53.399958, -36.127139], 
                               'SN2012ht':[163.344792, 16.776361], 
                               'SN2013dy':[334.573333, 40.569333], 
                               'SN2015F':[114.06567, -69.50639],# NEW,
                               "SN1980N":[50.751333, -37.21386],
                               "SN1981D":[50.659917, -37.232722],
                               "SN1989B":[170.058049, 13.005248],
                               "SN1994D":[188.509979, 7.701583],
                               "SN1998bu":[161.691792, 11.835306],
                               "SN2006dd":[50.673417, -37.203611],
                               "SN2007on":[54.712496, -35.575311],
                               "SN2011iv":[54.713958, -35.592222],
                                   },
                                  index=["ra","dec"]).T


def load_riess_target(sample="riess2016"):
    """ Load the list of target from Riess et al. 2016 """
    if sample == "riess2016":
        filepath = os.path.join(_SOURCEDIR,"data/riess2016_table5.csv")
    else:
        raise NotImplementedError(f"Only the riess2016 sample implemented ; {sample} given")
    
    target_ = pandas.read_csv(filepath , sep=",", index_col="SN")
    targets = target_.join(COORDINATES.loc[target_.index])
    targets.sort_index(inplace=True)
    return targets

def load_freedman_target(sample="freedman2019", which="table3"):
    """ """
    if sample == "freedman2019":
        filepath = os.path.join(_SOURCEDIR, f"data/freedman2019_{which}.csv")
    else:
        raise NotImplementedError(f"Only the freedman2019 sample implemented ; {sample} given")

    target_ = pandas.read_csv(filepath, sep=",", index_col="SN")
    targets = target_.join(COORDINATES.loc[target_.index])
    targets.sort_index(inplace=True)
    return targets

def load_riess_hf(cepheid_sn=True, avoid=None, sample="riess2016"):
    """ """
    if sample:
        filepath = os.path.join(_SOURCEDIR,"data/riess_2016_clean.csv")
    else:
        raise NotImplementedError(f"Only the riess2016 sample implemented ; {sample} given")
    
    datariess = pandas.read_csv( filepath )
    datariess["snname"] = ["SN"+name for name in datariess["CID"]]
    datariess.set_index("snname", inplace=True)
    if cepheid_sn:
        return datariess[datariess.index.isin([name for name in RIESS_TARGETS.index if not (avoid is not None and name in avoid) ])]
    return  datariess

def surcharge_with_anchoringdata(dataframe):
    """ """
    riess_csn = load_riess_hf( avoid="SN2011fe")
    # "SN2011fe issue fixing"
    datasnsh0es = dataframe.join(riess_csn)
    datasnsh0es.loc["SN2011fe", ["x1","x1ERR", "c","cERR"]] = -0.21, 0.07, -0.066, 0.021
    return datasnsh0es

def get_fullriess_anchoringdata():
    """ """
    return surcharge_with_anchoringdata( load_riess_target() )
    


# ======================= #
#                         #
#                         #
#  Special Data           #
#                         #
#                         #
# ======================= #
def _panstarrs_filename_(name, band):
    """ """
    return os.path.join(DATAPATH,"panstarrs","cutouts",f"{name}/{name}_panstarrs_{band}.fits" )

def get_panstarrs_files(name, bands=["g","r","i"], verbose=True):
    """ """
    fileout = {}
    for band in bands:
        filename = _panstarrs_filename_(name, band)
        if os.path.exists(filename):
            fileout[band] = filename
        else:
            if verbose:
                warnings.warn(f"no file found for {name} band {band} ({filename} not found)")
            fileout[band] = None
            
    return fileout

# ======================= #
#                         #
#                         #
#  CLASSES                #
#                         #
#                         #
# ======================= #
