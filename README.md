# WOW!
Wavelets Optimized Whitening
___

## Installation

Prior to installing WOW, the following packages must be installed from their respective GitHub repositories: 

 * [eui_selektor_client](https://github.com/gpelouze/eui_selektor_client) (lightweight client for the EUI Selektor)
 * [rectify](https://github.com/frederic-auchere/rectify) (geometric remapping)

Then, after cloning the present repository:

```shell
pip install .
```

will take care of the other dependencies. Or if you want to be able to edit & develop (requires reloading the package)

```shell
pip install -e .
```

The installation process will create a `wow` executable that can be run from the command line of the virtual environment (more below):

## Usage

The package provides two ways of processing images with the WOW! algorithm:
* The `wow` function that can be used in Python programs
* The `wow` executable that can be run from the command line

### In a Python program

An example is given in the `example.py` file. In essence:
```python
from astropy.io import fits
from wow import wow

sample_file = r'sample_data\solo_L2_eui-hrieuv174-image_20220317T032000234_V01.fits'
image = fits.getdata(sample_file)

# Basic WOW, non-bilateral, non de-noised. Fast.
wow_image = wow(image)
# Edge-aware, slower
bilateral_wow_image = wow(image, bilateral=1) 
# Same with de-noising
denoised_bilateral_wow_image = wow(image, denoise_coefficients=[5, 1], bilateral=1)  
```

### The wow executable

The `wow` executable can be called from the command line to produce movies from a sequence of files, directories or glob patterns. Help on the available parameters can be obtained with

```shell
wow --help

usage: WOW! [-h] (--source SOURCE | --selektor Selektor query [Selektor query ...] | --ascii Input ASCII file) [-o OUTPUT] [-d DENOISE [DENOISE ...]] [-w WEIGHTS [WEIGHTS ...]] [-nb] [-ns N_SCALES] [-gw GAMMA_WEIGHT] [-g GAMMA]
            [-nw] [-t] [-roi ROI ROI ROI ROI] [-r REGISTER] [-nu] [-cs] [-ne] [-c] [-fps FRAME_RATE] [-crf CRF] [-np N_PROCS] [-nc] [-nl] [-fn FIRST_N] [-i INTERVAL INTERVAL] [-rb REBIN] [-rt ROTATE] [-tf]

Processes a sequence of files with Wavelets Optimized Whitening and encodes the frames to video.

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE       List of files, directories or glob patterns
  --selektor Selektor query [Selektor query ...]
                        Queries Selektor for EUI observations
  --ascii Input ASCII file
                        ASCII file containing list of input files
  -o OUTPUT, --output OUTPUT
                        Output filename. Frames are saved in its base directory.
  -d DENOISE [DENOISE ...], --denoise DENOISE [DENOISE ...]
                        De-noising coefficients
  -w WEIGHTS [WEIGHTS ...], --weights WEIGHTS [WEIGHTS ...]
                        Synthesis weights
  -nb, --no_bilateral   Do not use edge-aware (bilateral) transform
  -ns N_SCALES, --n_scales N_SCALES
                        Number of wavelet scales
  -gw GAMMA_WEIGHT, --gamma_weight GAMMA_WEIGHT
                        Weight of gamma-stretched image
  -g GAMMA, --gamma GAMMA
                        Gamma exponent
  -nw, --no_whitening   Do not apply whitening (WOW!)
  -t, --temporal        Applies temporal de-noising and/or whitening
  -roi ROI ROI ROI ROI  Region of interest [bottom left, top right corners]
  -r REGISTER, --register REGISTER
                        Order of polynomial used to fit the header data to register the frames.
  -nu, --north_up       Rotate images north up
  -cs, --center_sun     Center solar disk
  -ne, --no_encode      Do not encode the frames to video
  -c, --cleanup         Cleanup frames after encoding
  -fps FRAME_RATE, --frame-rate FRAME_RATE
                        Number of frames per second
  -crf CRF              FFmpeg crf quality parameter
  -np N_PROCS, --n_procs N_PROCS
                        Number of processors to use. By default, uses 1 or the maximum available -1
  -nc, --no-clock       Do not inset clock
  -nl, --no-label       Do not inset time stamp & label
  -fn FIRST_N, --first_n FIRST_N
                        Process only the first N frames
  -i INTERVAL INTERVAL, --interval INTERVAL INTERVAL
                        Percentile to use for scaling
  -rb REBIN, --rebin REBIN
```


#### Movie from images in a directory

All images in the my/directory folder
```shell
wow --source my/directory -o movie.mp4
```
Only some files
```shell
wow --source my/directory/*20221002*.fits -o movie.mp4
```



#### Using Selektor queries

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

Note that the DATE-BEG FITS keyword indicates the beginning of an exposure. The *start* and *end* suffixes are added to *date_begin* to form the *date_begin_start* and *date_begin_end* parameters, which can be confusing. 

When passed the `-- selektor [query]` option, the `wow` executable uses Selektor's filtering capability to obtain, *e.g.* a list of FSI 174 image between two given dates, while excluding too short exposure times. Note that `eui_selektor_client` queries the EUI archive but **does not** download the data itself. It is up to the user to make sure that the data is available in a directory tree organized by `YYYY/MM/DD`. An environment variable named `EUI_ARCHIVE_DATA_PATH` must point at the root of the data tree. For example, the file `solo_L1_eui-hrieuv174-image_20220317T032000234_V01.fits` must be located in the `EUI_ARCHIVE_DATA_PATH/L1/2022/03/17` directory:

```shell
EUI_ARCHIVE_DATA_PATH
├─ L1
│  ├─ 2022/
│  │  ├─ 03
│  │  │  ├─ 17
│  │  │  │  ├─ solo_L1_eui-hrieuv174-image_20220317T032000234_V01.fits
```

#### From an ASCII file

A movie can be assembled from a list of file stored as individual lines in an ascii file. Full paths must be provided.  

```shell
wow --ascii file.txt -o movie.mp4
```

#### Examples

Queries selektor to create a video file in the movie directory from all the FSI 304 images from 2022-10-01 to 2022-10-30, excluding exposures shorter than 1 second (note that `2022-10-30` means `2022-10-30T00:00:00`):

```shell
wow --selektor detector[]:FSI wavelnth[]:304 date_begin_start:2022-10-01 date_begin_end:2022-10-30 image_size_min:3072 xposure_min:1 -o movie.mp4
```
Use the HRI_EUV data from 2022-10-19T00:00:00 to 2022-10-19T19:00:00, excluding exposures shorter than 1 second:

```shell
wow --selektor detector[]:HRI_EUV date_begin_start:2022-10-19 date_begin_end:2022-10-19 date_begin_end_hour:19 xposure_min:1 -o output/movie.mp4
```

# References

[Auchère, F., Soubrié, E., Pelouze, G. and Buchlin, É. 2022, *Image Enhancement With Wavelets Optimized Whitening*, A&A, 670, id.A66](https://www.aanda.org/articles/aa/pdf/2023/02/aa45345-22.pdf)
