# WOW

## Installation

It is recommended that this package be installed in a dedicated virtual environment, *e.g.* with 

```shell
python -m venv /path/to/new/virtual/environment
```

The following packages must be installed from their respective Github repositories: 

 * [eui_selektor_client](https://github.com/gpelouze/eui_selektor_client) (lightweight client for the EUI Selektor)
 * [watroo](https://github.com/frederic-auchere/wavelets) (à trous wavelets transforms)
 * [rectify](https://github.com/frederic-auchere/rectify) (geometric remapping)

Then:

```shell
pip install .
```

will install the other dependencies from PyPi. Or if you want to be able to edit & develop (requires reloading the package)

```shell
pip install -e .
```

The installation process will create the wow executable that can be run from the command line of the virtual environment (more below in the Usage section):

## Usage

The `wow` executable can be called from the command line to produce movies from sequence of files, directories or glob patterns.

### Using Selektor queries

[`eui_selektor_client`](https://github.com/gpelouze/eui_selektor_client) is a lightweight client to the [EUI Selektor](EUI Selektor) tool (password protected). Note that `eui_selektor_client` is used to query the EUI archive, but not to download the data itself.

When passed the `-- selektor [query]` option, `wow` uses Selektor's filtering capability to obtain, *e.g.* a list of FSI 174 image from two given dates, including a given binning level, and too short exposure times, but it is then up to the user to make sure that the data is available in a directory tree organized by `YYYY/MM/DD`. An environment variable named `EUI_ARCHIVE_DATA_PATH`environment variable must point at the root of the data tree. For example, the file `solo_L2_eui-hrieuv174-image_20220317T032000234_V01.fits` must be located in the `EUI_ARCHIVE_DATA_PATH/L2/2022/03/17` directory.

The query passed as value to the `--selektor` argument must be of the form `parameter1:value1 parameter2:value2`. The most useful query parameters and possible values are summarized below. Note that some parameters do require the [ ] brackets, e.g. `detector[]`

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

#### Examples

```python
wow --selektor detector[]:FSI wavelnth[]:304 date_begin_start:2022-10-01 date_begin_end:2022-10-30 image_size_min:3072 xposure_min:1 -o ~/movie/movie -d 5 1 -gw 0.995 -fps 24 -g 3.5 -r 1 -np 40 -i 99.99 -roi 0 0 3040 3072
```