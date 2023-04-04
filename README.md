# Relative Calibration Adjusment technique

The Relative Clutter Adjustment (RCA) technique is used for radar calibration monitoring, by estimating and correcting for unwanted "clutter" signals that can arise from various sources such as buildings, mountains, and trees. This repository contains code for performing RCA on weather radar data, as well as an example Jupyter notebook to demonstrate the technique.

## Libraries needed:

- numpy
- pandas 
- netCDF4
- xarray
- dask
- pyodim

These libraries can be installed using pip:
```
pip install numpy pandas netCDF4 xarray dask
```

In addition, you will need to install the cluttercal and pyodim libraries from Github:
```
pip install git+https://github.com/vlouf/cluttercal.git
pip install git+https://github.com/vlouf/pyodim.git`
```

## Example Jupyter Notebook

An example Jupyter notebook is available in the `example` directory. This notebook demonstrates how to use the `cluttercal` library to perform RCA on radar data. The notebook provides step-by-step instructions for downloading a sample of radar data from the Australian weather radar network archive, computing the clutter mask, and extracting the RCA value using the clutter mask. Finally, the notebook uses Matplotlib to create a plot of the RCA value over the radar data.

## References

If you use this code in your research, please cite the following paper:

- Louf, V., A. Protat, R. A. Warren, S. M. Collis, D. B. Wolff, S. Raunyiar, C. Jakob, and W. A. Petersen, 2018: An integrated approach to weather radar calibration and monitoring using ground clutter and satellite comparisons. J. Atmos. Ocean. Technol., JTECH-D-18-0007.1, doi:10.1175/JTECH-D-18-0007.1. [http://journals.ametsoc.org/doi/10.1175/JTECH-D-18-0007.1]

## License

This library is open source and made freely available according to the below
text:

    Copyright 2020 Valentin Louf
    Copyright 2023 Commonwealth of Australia, Bureau of Meteorology

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

A copy of the license is also provided in the LICENSE file included with the
source distribution of the library.
