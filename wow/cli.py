import os
import sys
import argparse
from .wow import main
from eui_selektor_client import EUISelektorClient
from eui_selektor_client.cli import parse_query_args


def eui_file2path(file, archive_path=''):
    return os.path.join(archive_path, file[5:7], file[-27:-23], file[-23:-21], file[-21:-19], file)


def cli():
    """
    Command line interface to the wow movie processor
    wow --help
    :return:
    """
    parser = argparse.ArgumentParser(prog="WOW!",
                                     description="Processes a sequence of files with Wavelets Optimized "
                                                 "Whitening and encodes the frames to video.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source",
                              help="List of files, directories or glob patterns",
                              type=str)
    source_group.add_argument("--selektor",
                              metavar="Selektor query",
                              help="Queries Selektor for EUI observations", type=str, nargs="+")
    source_group.add_argument("--ascii",
                              metavar="Input ASCII file",
                              help="ASCII file containing list of input files", type=str)
    parser.add_argument("-o", "--output",
                        help="Output filename. Frames are saved in its base directory.",
                        type=str)
    parser.add_argument("-d", "--denoise",
                        help="De-noising coefficients",
                        default=[],
                        type=float,
                        nargs='+')
    parser.add_argument("-w", "--weights",
                        help="Synthesis weights",
                        default=[],
                        type=float,
                        nargs='+')
    parser.add_argument("-nb", "--no_bilateral",
                        help="Do not use edge-aware (bilateral) transform",
                        action='store_true')
    parser.add_argument("-ns", "--n_scales",
                        help="Number of wavelet scales",
                        default=None,
                        type=int)
    parser.add_argument("-gw", "--gamma_weight",
                        help="Weight of gamma-stretched image",
                        default=0,
                        type=float)
    parser.add_argument("-g", "--gamma",
                        help="Gamma exponent",
                        default=2, type=float)
    parser.add_argument("-nw", "--no_whitening", help="Do not apply whitening (WOW!)",
                        action='store_true')
    parser.add_argument("-t", "--temporal",
                        help="Applies temporal de-noising and/or whitening",
                        action='store_true')
    parser.add_argument("-roi",
                        help="Region of interest [bottom left, top right corners]",
                        type=int,
                        nargs=4)
    parser.add_argument("-r", "--register",
                        help="Order of polynomial used to fit the header data to register the frames.",
                        type=int,
                        default=0)
    parser.add_argument("-nu", "--north_up",
                        help="Rotate images north up",
                        action="store_true")
    parser.add_argument("-cs", "--center_sun",
                        help="Center solar disk",
                        action="store_true")
    parser.add_argument("-ne", "--no_encode",
                        help="Do not encode the frames to video",
                        action='store_true')
    parser.add_argument("-c", "--cleanup",
                        help="Cleanup frames after encoding",
                        action='store_true')
    parser.add_argument("-fps", "--frame-rate",
                        help="Number of frames per second",
                        default=12,
                        type=float)
    parser.add_argument("-crf",
                        help="FFmpeg crf quality parameter",
                        default=22,
                        type=int)
    parser.add_argument("-np", "--n_procs",
                        help="Number of processors to use. By default, uses 1 or the maximum available -1",
                        default=0, type=int)
    parser.add_argument("-nc", "--no-clock",
                        help="Do not inset clock",
                        action='store_true')
    parser.add_argument("-nl", "--no-label",
                        help="Do not inset time stamp & label ",
                        action='store_true')
    parser.add_argument("-fn", "--first_n",
                        help="Process only the first N frames",
                        type=int)
    parser.add_argument("-i", "--interval",
                        help="Percentile to use for scaling",
                        default=[0.1, 99.9],
                        nargs=2,
                        type=float)
    parser.add_argument("-im", "--interval-margin",
                        help="Factor applied to interval top boundary",
                        default=1,
                        type=float)
    parser.add_argument("-rb", "--rebin",
                        help="binning factor",
                        default=1,
                        type=int)
    parser.add_argument("-rt", "--rotate",
                        help="Counter-clockwise 90 / 180 / 270° rotation",
                        default=0,
                        type=int)
    parser.add_argument("-tf", "--to_fits",
                        help="Save to fits",
                        action='store_true')

    args = parser.parse_args()

    if args.selektor:
        client = EUISelektorClient()
        query = parse_query_args(args.selektor)
        res = client.search_nolimit(query)
        if res is not None:
            archive_path = os.getenv('EUI_ARCHIVE_DATA_PATH')
            if archive_path is None:
                archive_path = ''
                print('Warning: undefined EUI_ARCHIVE_DATA_PATH')
            selektor_files = [eui_file2path(os.path.basename(f), archive_path) for f in res['filepath']]
            n_selektor_files = len(selektor_files)
            local_files = [f for f in selektor_files if os.path.exists(f)]
            n_local_files = len(local_files)
            if n_local_files != n_selektor_files:
                print(f'{n_selektor_files - n_local_files} files found by Selektor not found locally')
            if n_local_files == 0:
                print('No files found locally')
                sys.exit(1)
            args.source = local_files
        else:
            sys.exit(1)
    elif args.ascii:
        with open(args.ascii, 'r') as f:
            args.source = [line.rstrip() for line in f]

    main(**vars(args))
