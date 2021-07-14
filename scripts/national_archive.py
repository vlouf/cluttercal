"""
Radar calibration monitoring using ground clutter. Processing the Australian
National archive.

@creator: Valentin Louf <valentin.louf@bom.gov.au>
@institution: Monash University and Bureau of Meteorology
@date: 14/07/2021

.. autosummary::
    :toctree: generated/

    buffer
    check_reflectivity
    check_rid
    extract_zip
    gen_cmask
    get_radar_archive_file
    remove
    savedata
    main
"""
import gc
import os
import sys
import zipfile
import argparse
import warnings
import traceback

from typing import List, Tuple, Any

import crayons
import numpy as np
import pandas as pd
import dask.bag as db

import cluttercal


def buffer(infile: str, cmask: str, refl_name: str) -> Tuple[Any, float]:
    """
    Buffer function to catch and kill errors.

    Parameters:
    ===========
    infile: str
        Input radar file.

    Returns:
    ========
    dtime: np.datetime64
        Datetime of infile
    rca: float
        95th percentile of the clutter reflectivity.
    """
    try:
        dtime, rca = cluttercal.extract_clutter(infile, cmask, refl_name=refl_name)
    except ValueError:
        return None
    except Exception:
        print(infile)
        traceback.print_exc()
        return None

    return dtime, rca


def check_reflectivity(infile: str, refl_name: str) -> bool:
    """
    Check if the Radar file contains the uncorrected reflectivity field.

    Parameter:
    ==========
    infile: str
        Input radar file.

    Returns:
    ========
    is_good: bool
        True/False as the REFL_NAME field in data.
    """
    is_good = True
    try:
        _ = cluttercal.cluttercal.read_radar(infile, refl_name=refl_name)
    except KeyError:
        is_good = False
    except Exception:
        traceback.print_exc()
        is_good = False

    return is_good


def check_rid() -> bool:
    """
    Check if the Radar ID provided exists.
    """
    indir = f"/g/data/rq0/level_1/odim_pvol/{RID:02}"
    return os.path.exists(indir)


def extract_zip(inzip: str, path: str) -> List[str]:
    """
    Extract content of a zipfile inside a given directory.

    Parameters:
    ===========
    inzip: str
        Input zip file.
    path: str
        Output path.

    Returns:
    ========
    namelist: List
        List of files extracted from  the zip.
    """
    with zipfile.ZipFile(inzip) as zid:
        zid.extractall(path=path)
        namelist = [os.path.join(path, f) for f in zid.namelist()]
    return namelist


def gen_cmask(radar_file_list: List[str], outputfile: str, refl_name: str) -> None:
    """
    Generate the clutter mask for a given day and save the clutter mask as a
    netCDF.

    Parameters:
    ===========
    radar_file_list: list
        List radar files for the given date.
    outputfile: str
        Output file name.

    Returns:
    ========
    outpath: str
        Output directory for the clutter masks.
    """
    if os.path.isfile(outputfile):
        print("Clutter masks already exists. Doing nothing.")
        return None

    try:
        cmask = cluttercal.clutter_mask(
            radar_file_list,
            refl_name=refl_name,
            refl_threshold=REFL_THLD,
            max_range=20e3,
            freq_threshold=50,
            use_dask=True,
        )
    except Exception:
        traceback.print_exc()
        return None

    if cmask is None:
        raise ValueError(f"!!! COULD NOT CREATE CLUTTER MAP FOR!!!")
    else:
        cmask.to_netcdf(outputfile)

    return None


def get_radar_archive_file(date: pd.Timestamp) -> str:
    """
    Return the archive containing the radar file for a given radar ID and a
    given date.

    Parameters:
    ===========
    date: datetime
        Date.

    Returns:
    ========
    file: str
        Radar archive if it exists at the given date.
    """
    datestr = date.strftime("%Y%m%d")
    file = f"/g/data/rq0/level_1/odim_pvol/{RID:02}/{date.year}/vol/{RID:02}_{datestr}.pvol.zip"
    if not os.path.exists(file):
        return None

    return file


def remove(flist: List[str]) -> None:
    """
    Remove file if it exists.
    """
    flist = [f for f in flist if f is not None]
    for f in flist:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    return None


def savedata(df, date, path: str) -> None:
    """
    Save the output data into a CSV file compatible with pandas.

    Parameters:
    ===========
    df: pd.Dataframe
        RCA timeserie to be saved.
    date:
        Date of processing.
    path: str
        Output directory.
    """
    datestr = date.strftime("%Y%m%d")

    path = os.path.join(path, "rca", str(RID))
    os.makedirs(path, exist_ok=True)

    outfilename = os.path.join(path, f"rca.{RID}.{datestr}.csv")
    df.to_csv(outfilename)
    print(crayons.green(f"Found {len(df)} hits for {datestr}."))
    print(crayons.green(f"Results saved in {outfilename}."))

    return None


def main(date_range) -> None:
    """
    Loop over dates:
    1/ Unzip archives.
    2/ Generate clutter mask for given date.
    3/ Generate composite mask.
    4/ Get the 95th percentile of the clutter reflectivity.
    5/ Save data for the given date.
    6/ Remove unzipped file and go to next iteration.

    Parameters:
    ===========
    date_range: Iter
        List of dates to process
    """
    print(crayons.green(f"RCA processing for radar {RID}."))
    print(crayons.green(f"Between {START_DATE} and {END_DATE}."))
    print(crayons.green(f"Data will be saved in {OUTPATH}."))

    prefix = f"{RID}_"
    for date in date_range:
        # Get zip archive for given radar RID and date.
        zipfile = get_radar_archive_file(date)
        if zipfile is None:
            print(crayons.yellow(f"No file found for radar {RID} at date {date}."))
            continue

        # Unzip data/
        namelist = extract_zip(zipfile, path=ZIPDIR)
        refl_name = None
        for name in REFL_NAME:
            if check_reflectivity(namelist[0], refl_name=name):
                refl_name = name
                break

        if refl_name is None:
            print("No valid reflectivity fields found at date {date}.")
            continue

        print(crayons.yellow(f"{len(namelist)} files to process for {date}."))

        # Generate clutter mask for the given date.
        datestr = date.strftime("%Y%m%d")
        outpath = os.path.join(OUTPATH, "cmasks", str(RID))
        os.makedirs(outpath, exist_ok=True)
        output_maskfile = os.path.join(outpath, prefix + f"{datestr}.nc")
        try:
            gen_cmask(namelist, output_maskfile, refl_name=refl_name)
        except ValueError:
            pass

        # Generate composite mask.
        try:
            cmask = cluttercal.composite_mask(date, timedelta=7, indir=outpath, prefix=prefix)
        except ValueError:
            cmask = cluttercal.single_mask(date, indir=outpath, prefix=prefix)

        # Extract the clutter reflectivity for the given date.
        arglist = [(f, cmask, refl_name) for f in namelist]
        bag = db.from_sequence(arglist).starmap(buffer)
        rslt = bag.compute()

        saved = False
        if rslt is not None:
            rslt = [r for r in rslt if r is not None]
            if len(rslt) != 0:
                ttmp, rtmp = zip(*rslt)
                rca = np.array(rtmp)
                dtime = np.array(ttmp, dtype="datetime64")

                if len(rca) != 0:
                    df = pd.DataFrame({"rca": rca}, index=dtime)
                    savedata(df, date, path=OUTPATH)
                    saved = True

        if saved:
            print(crayons.green(f"Radar {RID} processed and RCA saved."))
        else:
            print(crayons.yellow(f"No data for radar {RID} for {date}."))

        # Removing unzipped files, collecting memory garbage.
        remove(namelist)
        gc.collect()

    return None


if __name__ == "__main__":
    parser_description = "Relative Calibration Adjustment (RCA) - Monitoring of clutter radar reflectivity."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "-s", "--start-date", dest="start_date", type=str, help="Left bound for generating dates.", required=True
    )
    parser.add_argument(
        "-e", "--end-date", dest="end_date", type=str, help="Right bound for generating dates.", required=True
    )
    parser.add_argument(
        "-r",
        "--rid",
        dest="rid",
        type=int,
        required=True,
        help="The individual radar Rapic ID number for the Australian weather radar network.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="/scratch/kl02/vhl548/s3car-server/cluttercal/",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        dest="refl_thld",
        type=int,
        default=45,
        help="Clutter detection minimal reflectivity threshold in dBZ.",
    )

    # Parser arguments
    args = parser.parse_args()
    RID: int = args.rid
    START_DATE: str = args.start_date
    END_DATE: str = args.end_date
    OUTPATH: str = args.output
    REFL_THLD: int = args.refl_thld

    # Global parameters
    REFL_NAME: Tuple[str, str] = ("TH", "DBZH")
    ZIPDIR: str = "/scratch/kl02/vhl548/unzipdir/"

    if not check_rid():
        parser.error("Invalid Radar ID.")
        sys.exit()

    try:
        date_range = pd.date_range(START_DATE, END_DATE)
        if len(date_range) == 0:
            parser.error("End date older than start date.")
    except Exception:
        parser.error("Invalid dates.")
        sys.exit()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(date_range)
