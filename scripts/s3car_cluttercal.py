"""
Quality control of Radar calibration monitoring using ground clutter

@creator: Valentin Louf <valentin.louf@bom.gov.au>
@project: s3car-server
@institution: Bureau of Meteorology
@date: 23/10/2020

    check_reflectivity
    driver
    mkdir
    main
"""
# Python Standard Library
import os
import sys
import glob
import argparse
import datetime
import traceback
import warnings

from typing import Tuple

# Other libraries.
import netCDF4
import numpy as np
import pandas as pd
import dask.bag as db

import cluttercal
from cluttercal.cluttermask import EmptyFieldError


def check_reflectivity(infile: str) -> bool:
    """
    Check for the presence of the Uncorrected Reflectivity fields in the ODIM
    h5 dataset. By convention the field name is TH.

    Parameter:
    ==========
    infile: str
        Input ODIM H5 file.
    Returns:
    ========
    True/False presence of the uncorrected reflectivity.
    """
    with netCDF4.Dataset(infile) as ncid:
        groups = ncid['/dataset1'].groups.keys()
        var = []
        for group in groups:
            if "data" not in group:
                continue
            name = ncid[f'/dataset1/{group}/what'].getncattr('quantity')
            var.append(name)

    if 'TH' in var:
        return True
    else:
        return False


def driver(infile: str, cmask: str) -> Tuple[]:
    """
    Buffer function to catch and kill errors about missing Sun hit.

    Parameters:
    ===========
    infile: str
        Input radar file.

    Returns:
    ========
    rslt: pd.DataFrame
        Pandas dataframe with the results from the solar calibration code.
    """
    try:
        dtime, rca = cluttercal.extract_clutter(infile, cmask, refl_name=REFL_NAME)
    except ValueError:
        return None
    except Exception:
        print(infile)
        traceback.print_exc()
        return None

    return dtime, rca


def mkdir(path: str) -> None:
    """
    Make directory if it does not already exist.
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    return None


def main():
    """
    Structure:
    1/ Create output directories (if does not exists)
    2/ Check if ouput exists (doing nothing if it does).
    3/ Check if input directories exists
    4/ Processing solar calibration
    5/ Saving output data.
    """
    rid, date = RID, DATE

    # Create output directories and check if output file exists
    outpath = os.path.join(OUTPUT_DATA_PATH, "rca")
    mkdir(outpath)
    outpath = os.path.join(outpath, str(rid))
    mkdir(outpath)    
    outfilename = os.path.join(outpath, f"rca.{rid}.{date}.csv")
    if os.path.isfile(outfilename):
        print("Output file already exists. Doing nothing.")
        return None

    # Input directory checks.
    input_dir = os.path.join(VOLS_ROOT_PATH, str(rid))
    if not os.path.exists(input_dir):
        print(f"RAPIC ID: {RID} not found in {VOLS_ROOT_PATH}.")
        return None

    input_dir = os.path.join(input_dir, date)
    if not os.path.exists(input_dir):
        print(f"Date: {DATE} not found in {VOLS_ROOT_PATH} for radar {RID}.")
        return None

    input_dir = os.path.join(input_dir, "*.h5")
    flist = sorted(glob.glob(input_dir))
    if len(flist) == 0:
        print(f"No file found for radar {RID} at {DATE}.")
        return None

    goodfiles = [*map(check_reflectivity, flist)]
    if not any(goodfiles):
        print(f"The uncorrected reflectivity field is not present for radar {RID}.")
        return None
    flist = [f for f, g in zip(flist, goodfiles) if g is True]
    print(f"Found {len(flist)} files with the uncorrected reflectivity for radar {RID} for date {DATE}.")

    # Find clutter and save mask.
    outpath_cmask = os.path.join(OUTPUT_DATA_PATH, "cmasks")
    mkdir(outpath_cmask)
    outpath_cmask = os.path.join(outpath_cmask, f"{RID}")
    mkdir(outpath_cmask)
    outputfile_cmask = os.path.join(outpath_cmask, f"{RID}_{DATE}.nc")

    cluttercal.clutter_mask(flist, output=outputfile_cmask, refl_name=REFL_NAME, refl_threshold=REFL_THLD, use_dask=False)

    # Generate composite clutter mask.
    try:
        cmask = cluttercal.composite_mask(DTIME, timedelta=7, indir=outpath_cmask, prefix=f"{RID}_")
    except ValueError:
        # single mask
        cmask = cluttercal.single_mask(outputfile_cmask)

    # Processing RCA
    arglist = [(f, cmask) for f in flist]
    bag = db.from_sequence(arglist).starmap(driver)
    rslt = bag.compute()
    dataframe_list = [r for r in rslt if r is not None]
    if len(dataframe_list) == 0:
        print(f"No results for date {date}.")
        return None
    else:
        ttmp, rtmp = zip(*rslt)
        rca = np.array(rtmp)
        dtime = np.array(ttmp, dtype="datetime64")
        df = pd.DataFrame({"rca": rca}, index=dtime)
        if len(df) != 0:
            df.to_csv(outfilename, float_format="%g")
            print(f"Results saved in {outfilename}.")

    return None


if __name__ == "__main__":
    VOLS_ROOT_PATH = "/srv/data/s3car-server/vols"
    REFL_NAME = "total_power"
    REFL_THLD = 40  # Clutter Reflectivity threshold.

    parser_description = (
        "Quality control of antenna alignment and receiver calibration using the sun."
    )
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "-r",
        "--rid",
        dest="rid",
        type=int,
        required=True,
        help="Radar RAPIC ID number.",
    )
    parser.add_argument(
        "-d",
        "--date",
        dest="date",
        type=str,
        help="Value to be converted to Timestamp (str).",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output",
        default="/srv/data/s3car-server/cluttercal/data",
        type=str,
        help="Directory for output data.",
    )

    args = parser.parse_args()
    RID = args.rid
    DATE = args.date
    OUTPUT_DATA_PATH = args.output
    try:
        # 2 advantages: check if provided dtime is valid and turns it into a timestamp object.
        DTIME = pd.Timestamp(DATE)
    except Exception:
        traceback.print_exc()
        sys.exit()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
