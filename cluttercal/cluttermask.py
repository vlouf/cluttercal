"""
Generate clutter mask

title: cluttermask.py
author: Valentin Louf
email: valentin.louf@bom.gov.au
institution: Monash University and Bureau of Meteorology
date: 14/07/2021

.. autosummary::
    :toctree: generated/

    read_radar
    get_metadata
    find_clutter_pos
    clutter_mask
"""
import os
import warnings
import traceback

from typing import Tuple, List, Dict

import pyart
import pyodim
import numpy as np
import xarray as xr
import dask.bag as db


def read_radar(infile: str, refl_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read radar data using pyodim or pyart depending on the radar format.

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
    use_pyodim = False
    if infile.lower().endswith((".h5", ".hdf", ".hdf5")):
        try:
            r = pyodim.read_odim(infile)
            radar = r[0].compute()
            use_pyodim = True
        except Exception:
            radar = pyart.aux_io.read_odim_h5(infile, file_field_names=True, include_fields=[refl_name])
    else:
        radar = pyart.io.read(infile, include_fields=[refl_name])

    try:
        _ = radar[refl_name].values
    except KeyError:
        raise KeyError(f"Problem with {os.path.basename(infile)}: uncorrected reflectivity not present.")

    if use_pyodim:
        r = radar.range.values
        azi = np.round(radar.azimuth.values % 360).astype(int)
        refl = radar[refl_name].values
        try:
            refl = refl.filled(np.NaN)
        except Exception:
            pass
    else:
        elev = radar.elevation["data"]
        lowest_tilt = np.argmin([elev[i][0] for i in radar.iter_slice()])
        sl = radar.get_slice(lowest_tilt)

        r = radar.range["data"]
        azi = np.round(radar.azimuth["data"][sl] % 360).astype(int)
        refl = radar.fields[refl_name]["data"][sl].filled(np.NaN)

    return r, azi, refl


def get_metadata(infile: str) -> Dict:
    """
    Extract general metadata from radar file.

    Parameter:
    ==========
    infile: str
        Input file name

    Returns:
    ========
    attrs: Dict
        Metadata dictionnary.
    """
    use_pyodim = False
    if infile.lower().endswith((".h5", ".hdf", ".hdf5")):
        try:
            r = pyodim.read_odim(infile)
            radar = r[0].compute()
            use_pyodim = True
        except Exception:
            radar = pyart.aux_io.read_odim_h5(infile, file_field_names=True)
    else:
        radar = pyart.io.read(infile)

    if use_pyodim:
        attrs = radar.attrs
    else:
        attrs = radar.metadata

    try:
        for k, v in attrs.items():
            attrs[k] = str(v)
    except Exception:
        return None
    return attrs


def find_clutter_pos(
    infile: str, refl_name: str = "TH", refl_threshold: float = 40, max_range: float = 20e3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find high reflectivity pixels in the lowest tilt of the radar scan.

    Parameter:
    ==========
    infile: str
        Input radar file
    refl_name: str
        Uncorrected reflectivity field name.
    refl_threshold: float
        Minimum reflectivity value threshold.
    max_range: int
        Maximum range (in m) threshold.

    Returns:
    ========
    rclutter: ndarray
        Range value of clutter pixels.
    aziclutter: ndarray
        Azimuth value of clutter pixels.
    zclutter: ndarray
        Reflectivity value of clutter pixels.
    """
    try:
        r, azi, refl = read_radar(infile, refl_name)
    except Exception:
        traceback.print_exc()
        return None

    R, A = np.meshgrid(r, azi)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos = (R < max_range) & (refl > refl_threshold)

    rclutter = 1000 * (R[pos] / 1e3).astype(int)
    aziclutter = A[pos]
    zclutter = np.round(2 * refl[pos]) / 2

    return rclutter, aziclutter, zclutter


def clutter_mask(
    radar_file_list: List,
    output: str = None,
    refl_name: str = "TH",
    refl_threshold: float = 50,
    max_range: float = 20e3,
    freq_threshold: float = 50,
    use_dask: bool = True,
) -> xr.Dataset:
    """
    Extract the clutter and compute the RCA value.

    Parameters:
    ===========
    radar_file_list: str
        List of radar files.
    output: str
        If output is defined, will saved the mask, otherwise it will return the mask.
    refl_name: str
        Uncorrected reflectivity field name.
    refl_threshold: float
        Minimum reflectivity value threshold.
    max_range: int
        Maximum range (in m) threshold.
    freq_threshold: int
        Minimum clutter frequency threshold in %.
    use_dask: bool
        Use dask multiprocessing to parse the input list of radar files.

    Returns:
    ========
    dset: xr.Dataset
        Clutter mask.
    """
    argslist = []
    for f in radar_file_list:
        argslist.append((f, refl_name, refl_threshold, max_range))

    if use_dask:
        bag = db.from_sequence(argslist).starmap(find_clutter_pos)
        rslt = bag.compute()
    else:
        rslt = [find_clutter_pos(d) for d in radar_file_list]

    rslt = [r for r in rslt if r is not None]
    if len(rslt) == 0:
        raise ValueError("No Clutter detected")
    freq_ratio = 100 / len(rslt)

    nr = int(max_range // 1000)
    na = 360

    cmask = np.zeros((len(rslt), na, nr))
    zmask = np.zeros((len(rslt), na, nr)) + np.NaN

    for idx, (r, a, refl) in enumerate(rslt):
        rpos = (r // 1000).astype(int)
        apos = a.astype(int) % 360
        cmask[idx, apos, rpos] = 1
        zmask[idx, apos, rpos] = refl
    zmask = np.ma.masked_invalid(zmask)

    arr = np.zeros((na, nr), dtype=np.int16)
    pos = (~np.ma.masked_less(freq_ratio * cmask.sum(axis=0), freq_threshold).mask) & (
        zmask.mean(axis=0) > refl_threshold
    )
    arr[pos] = 1

    if np.sum(arr) == 0:
        print("No Clutter detected. Not creating clutter mask.")
        return None

    dset = xr.Dataset(
        {
            "clutter_mask": (("azimuth", "range"), arr.astype(np.int16)),
            "azimuth": (("azimuth"), np.arange(na).astype(np.int16)),
            "range": (("range"), np.arange(nr).astype(np.int16)),
        }
    )

    dset.attrs = get_metadata(radar_file_list[0])
    dset.attrs["range_max"] = max_range
    dset.range.attrs = {"units": "km", "long_name": "radar_range"}
    dset.azimuth.attrs = {"units": "degrees", "long_name": "radar_azimuth"}
    dset.clutter_mask.attrs = {
        "units": "",
        "long_name": "clutter_mask",
        "description": "Clutter position in a coarse polar grid.",
    }

    if output is not None:
        dset.to_netcdf(output)
        print(f"Clutter mask {output} created.")
        return None

    return dset
