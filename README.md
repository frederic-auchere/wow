# WOW!
Wavelets Enhanced Whitening

## Installation

After cloning this repository, it is recommended to install this package and its dependencies in a fresh virtual environment, *e.g.* with

```shell
python -m venv wow/.venv
```
followed by `. .venv\bin\activate`(Unix) or `.venv\Scripts\activate` (Windows). 

The following packages must be installed from their respective GitHub repositories: 

 * [eui_selektor_client](https://github.com/gpelouze/eui_selektor_client) (lightweight client for the EUI Selektor)
 * [watroo](https://github.com/frederic-auchere/wavelets) (à trous wavelets transforms)
 * [rectify](https://github.com/frederic-auchere/rectify) (geometric remapping)

Then:

```shell
pip install .
```

will take care of the other dependencies. Or if you want to be able to edit & develop (requires reloading the package)

```shell
pip install -e .
```

The installation process will create a `wow` executable that can be run from the command line of the virtual environment (more below):

## Usage

The `wow` executable can be called from the command line to produce movies from a sequence of files, directories or glob patterns.

### Using Selektor queries

[`eui_selektor_client`](https://github.com/gpelouze/eui_selektor_client) is a lightweight client to the [EUI Selektor](https://www.sidc.be/EUI/data_internal/selektor) tool (password protected). The query passed as value to the `--selektor` argument must be of the form `--selektor parameter1:value1 parameter2:value2`. The most useful query parameters and possible values are summarized below. Note that some parameters do require the [ ] brackets, e.g. `detector[]:FSI`

| Parameter               | Possible values                                                           | FITS keyword  |
|-------------------------|---------------------------------------------------------------------------|---------------|
| level[]                 | 'L0', 'L1'                                                                | LEVEL         |
| image_size_min          | 320-3072                                                                  | NAXIS1=NAXIS2 |
| image_size_max          | 320-3072                                                                  | NAXIS1=NAXIS2 |
| binning[]               | 1, 2, 4                                                                   | BINNING       |
| detector[]              | 'FSI', 'HRI_EUV', 'HRI_LYA'                                               | DETECTOR      |
| wavelnth[]              | '174', '304', '1216'                                                      | WAVELNTH      |
| recstate[]              | 'on', 'off'                                                               | RECSTATE      |
| compress[]              | 'Lossless', 'Lossy-high quality', 'Lossy-strong', 'Lossy-extreme', 'None' | COMPRES       |
| gaincomb[]              | 'low-only', 'high-only', 'combined', 'both', 'other'                      | GAINCOMB      |
| imgtype[]               | 'solar image', 'LED image', 'dark image', 'occulted image'                | IMGTYPE       |
| xposure_min             | 0-7200                                                                    | XPOSURE       |
| xposure_max             | 0-7200                                                                    | XPOSURE       |
| date_begin_start        | YYYY-MM-DD                                                                | DATE-BEG      |
| date_begin_start_hour   | 0-23                                                                      | DATE-BEG      |
| date_begin_start_minute | 0-59                                                                      | DATE-BEG      |
| date_begin_end          | YYYY-MM-DD                                                                | DATE-BEG      |
| date_begin_end_hour     | 0-23                                                                      | DATE-BEG      |
| date_begin_end_minute   | 0-59                                                                      | DATE-BEG      |

When passed the `-- selektor [query]` option, the `wow` executable uses Selektor's filtering capability to obtain, *e.g.* a list of FSI 174 image between two given dates, while excluding too short exposure times. Note that `eui_selektor_client` queries the EUI archive but **does not** download the data itself. It is up to the user to make sure that the data is available in a directory tree organized by `YYYY/MM/DD`. An environment variable named `EUI_ARCHIVE_DATA_PATH` must point at the root of the data tree. For example, the file `solo_L2_eui-hrieuv174-image_20220317T032000234_V01.fits` must be located in the `EUI_ARCHIVE_DATA_PATH/L2/2022/03/17` directory:

```shell
EUI_ARCHIVE_DATA_PATH
├─ L2
│  ├─ 2022/
│  │  ├─ 03
│  │  │  ├─ 17
│  │  │  │  ├─ solo_L2_eui-hrieuv174-image_20220317T032000234_V01.fits
```

##
# Examples

Queries selektor to create a movie of FSI 304 images from 2022-10-01 to 2022-10-30, excluding exposures shorter than 1 second:
```shell
wow --selektor detector[]:FSI wavelnth[]:304 date_begin_start:2022-10-01 date_begin_end:2022-10-30 image_size_min:3072 xposure_min:1 -o movie
```

# References

Auchère, F., Soubrié, E., Pelouze, G., Buchlin, É. 2022, Image Enhancement With Wavelets Optimized Whitening, submitted to A&A