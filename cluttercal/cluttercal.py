"""
Compute the RCA.

title: rca.py
author: Valentin Louf
email: valentin.louf@bom.gov.au
institution: Monash University and Bureau of Meteorology
date: 01/06/2022

.. autosummary::
    :toctree: generated/

    read_radar
    composite_mask
    single_mask
    extract_clutter
"""
import os
import warnings
from typing import Tuple, Any

import pyodim
import numpy as np
import pandas as pd
import xarray as xr


def read_radar(infile: str, refl_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Read radar data.

    Parameters:
    ===========
    infile: str
        Input file
    refl_name: str
        Uncorrected reflectivity field name.

    Returns:
    ========
    radar: xr.Dataset
        Radar dataset, first elevation only.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        r = pyodim.read_odim(infile)
        radar = r[0].compute()
        try:
            _ = radar[refl_name].values
        except KeyError:
            raise KeyError(f"Problem with {os.path.basename(infile)}: uncorrected reflectivity not present.")

        r = radar.range.values
        azi = np.round(radar.azimuth.values % 360).astype(int)
        dtime = radar.time[0].values
        refl = radar[refl_name].values
        try:
            refl = refl.filled(np.NaN)
        except Exception:
            pass

    return r, azi, refl, dtime


def composite_mask(
    date: pd.Timestamp,
    timedelta: int = 7,
    indir: str = "compomask",
    prefix: str = "cpol_cmask_",
    freq_thrld: float = 0.9,
) -> np.ndarray:
    """
    Generate composite clutter mask.

    Parameters:
    -----------
    date: Timestamp
        Date of processing
    timedelta: int
        Time delta for the composite.
    indir: str
        Where clutter mask are stored
    prefix: str
        What is the clutter mask file prefix.
    freq_thrld: float
        Frequency threshold (0 < f < 1).

    Returns:
    --------
    mask: ndarray
        Clutter mask
    """

    def get_mask_list(date, timedelta, indir, prefix):
        drange = pd.date_range(date - pd.Timedelta(f"{timedelta}D"), date)
        flist = []
        for day in drange:
            file = os.path.join(indir, prefix + "{}.nc".format(day.strftime("%Y%m%d")))
            if os.path.isfile(file):
                flist.append(file)
        return flist

    flist = get_mask_list(date, timedelta, indir, prefix)
    if len(flist) == 1:
        composite = xr.open_dataset(flist[0]).clutter_mask.values
    else:
        cmask = [xr.open_dataset(f).clutter_mask.values for f in flist]
        cmaskarr = np.concatenate(cmask, axis=np.newaxis).reshape((len(flist), 360, 20))
        compo_freq = np.nansum(cmaskarr, axis=0) / len(flist)
        composite = compo_freq > freq_thrld
        if np.sum(composite) == 0:
            composite = compo_freq > 0.5
            if np.sum(composite) == 0:
                print(
                    f"Bad composite for {date} - Maximum clutter threshold found of {compo_freq.max()} for a threshold of {freq_thrld}."
                )
                composite = compo_freq != 0

    return composite


def single_mask(mask_file: str) -> np.ndarray:
    """
    Generate clutter mask.

    Parameters:
    -----------
    mask_file: input mask

    Returns:
    --------
    mask: ndarray
        Clutter mask
    """
    cmask = xr.open_dataset(mask_file).clutter_mask.values
    return cmask


def extract_clutter(infile: str, clutter_mask: np.ndarray, refl_name: str = "TH") -> Tuple[Any, float]:
    """
    Extract the clutter and compute the RCA value.

    Parameters:
    -----------
    infile: str
        Input radar file.
    clutter_mask: numpy.array(float)
        Clutter mask (360 deg x 20 km)
    refl_name: str
        Uncorrected reflectivity field name.

    Returns:
    --------
    dtime: np.datetime64
        Datetime of infile
    rca: float
        95th percentile of the clutter reflectivity.
    """
    # Radar data.
    r, azi, reflectivity, dtime = read_radar(infile, refl_name)
    refl = reflectivity[:, r < 20e3]
    zclutter = np.zeros_like(refl) + np.NaN

    r = r[r < 20e3]
    R, A = np.meshgrid(r, azi)
    R = (R // 1000).astype(int)
    A = (np.round(A) % 360).astype(int)

    # Mask.
    RC, AC = np.meshgrid(np.arange(20), np.arange(360))

    npos = np.where(clutter_mask)
    for ir, ia in zip(RC[npos], AC[npos]):
        pos = (R == ir) & (A == ia)
        zclutter[pos] = refl[pos]

    try:
        zclutter = zclutter[~np.isnan(zclutter)]
        rca = np.percentile(zclutter, 95)
    except IndexError:
        # Empty array full of NaN.
        raise ValueError("All clutter is NaN.")

    return dtime, rca
