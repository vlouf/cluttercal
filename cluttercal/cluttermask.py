"""
Generate clutter mask.

title: cluttermask.py
author: Valentin Louf
email: valentin.louf@bom.gov.au
institution: Monash University and Bureau of Meteorology
date: 11/02/2021
"""
import os
import warnings

import pyart
import numpy as np
import xarray as xr
import dask.bag as db


class EmptyFieldError(Exception):
    pass


def _read_radar(infile, refl_name):
    """
    Read input radar file

    Parameters:
    ===========
    radar_file_list: str
        List of radar files.
    refl_name: str
        Uncorrected reflectivity field name.

    Returns:
    ========
    radar: PyART.Radar
        Radar data.
    """
    try:
        if infile.lower().endswith(('.h5', '.hdf', '.hdf5')):
            radar = pyart.aux_io.read_odim_h5(infile, include_fields=[refl_name])
        else:
            radar = pyart.io.read(infile, include_fields=[refl_name])
    except Exception:
        print(f'Reading error of {infile}.')
        raise

    try:
        radar.fields[refl_name]
    except KeyError:
        del radar
        raise KeyError(f'Problem with {os.path.basename(infile)}: uncorrected reflectivity not present.')

    return radar


def clutter_mask(
    radar_file_list,
    output=None,
    refl_name="total_power",
    refl_threshold=50,
    max_range=20e3,
    freq_threshold=50,
    use_dask=True,
):
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
    def find_clutter_pos(infile: str):
        """
        Find high reflectivity pixels in the lowest tilt of the radar scan.

        Parameter:
        ==========
        infile: str
            Input radar file

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
            radar = _read_radar(infile, refl_name)
        except Exception:
            return None
        elev = radar.elevation['data']
        lowest_tilt = np.argmin([elev[i][0] for i in radar.iter_slice()])
        sl = radar.get_slice(lowest_tilt)

        r = radar.range["data"]
        azi = np.round(radar.azimuth["data"][sl] % 360).astype(int)
        refl = radar.fields[refl_name]["data"][sl].filled(np.NaN)
        R, A = np.meshgrid(r, azi)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pos = (R < max_range) & (refl > refl_threshold)

        rclutter = 1000 * (R[pos] / 1e3).astype(int)
        aziclutter = A[pos]
        zclutter = np.round(2 * refl[pos]) / 2

        del radar
        return rclutter, aziclutter, zclutter

    if use_dask:
        bag = db.from_sequence(radar_file_list).map(find_clutter_pos)
        rslt = bag.compute()
    else:
        rslt = [find_clutter_pos(d) for d in radar_file_list]

    rslt = [r for r in rslt if r is not None]
    if len(rslt) == 0:
        raise EmptyFieldError("No Clutter detected")
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

    arr = ((~np.ma.masked_less(freq_ratio * cmask.sum(axis=0), freq_threshold).mask) &
           (zmask.mean(axis=0) > refl_threshold))

    if np.sum(arr) == 0:
        print("No Clutter detected. Not creating clutter mask.")
        return None

    dset = xr.Dataset(
        {
            "clutter_mask": (("azimuth", "range"), arr),
            "azimuth": (("azimuth"), np.arange(na).astype(np.int16)),
            "range": (("range"), np.arange(nr).astype(np.int16)),
        }
    )

    radar = _read_radar(radar_file_list[0], refl_name)
    dset.attrs = radar.metadata
    dset.attrs['range_max'] = max_range
    dset.range.attrs = {"units": "km", "long_name": "radar_range"}
    dset.azimuth.attrs = {"units": "degrees", "long_name": "radar_azimuth"}
    dset.clutter_mask.attrs = {
        "units": "",
        "long_name": "clutter_mask",
        "description": "Clutter position in a coarse polar grid.",
    }

    del radar
    if output is not None:
        dset.to_netcdf(output)
        print(f'Clutter mask {output} created.')
        return None

    return dset
