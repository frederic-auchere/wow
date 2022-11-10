import os
import sys
import argparse
from .wow import main
from eui_selektor_client import EUISelektorClient
from eui_selektor_client.cli import parse_query_args


def eui_file2path(file, archive_path=''):
    return os.path.join(archive_path, file[5:7], file[-27:-23], file[-23:-21], file[-21:-19], file)


def cli():
    parser = argparse.ArgumentParser(prog="WOW!", description="Process sequence of files with Wavelets Optimized "
                                                              "Whitening and encodes the frames to video.")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--source", help="List of files, directories or glob patterns", type=str)
    source_group.add_argument("--selektor", help="Queries Selektor for EUI observations", type=str, nargs="+")
    parser.add_argument("-o", "--output_directory", help="Output directory", default='.', type=str)
    parser.add_argument("-d", "--denoise", help="Denoising coefficients", default=[], type=float, nargs='+')
    parser.add_argument("-nb", "--no_bilateral", help="Do not use edge-aware (bilateral) transform",
                        action='store_true')
    parser.add_argument("-ns", "--n_scales", help="Number of wavelet scales", default=None, type=int)
    parser.add_argument("-gw", "--gamma_weight", help="Weight of gamma-stretched image", default=0, type=float)
    parser.add_argument("-g", "--gamma", help="Gamma exponent", default=2, type=float)
    parser.add_argument("-nw", "--no_whitening", help="Do not apply whitening (WOW!)", action='store_true')
    parser.add_argument("-t", "--temporal", help="Applies temporal denoising and/or whitening", action='store_true')
    parser.add_argument("-roi", help="Region of interest [bottom left, top right corners]", type=int, nargs=4)
    parser.add_argument("-r", "--register", help="Uses header information to register the frames", type=int, default=2)
    parser.add_argument("-ne", "--no_encode", help="Do not encode the frames to video", action='store_true')
    parser.add_argument("-fps", "--frame-rate", help="Number of frames per second", default=12, type=float)
    parser.add_argument("-np", "--n_procs", help="Number of processors to use", default=0, type=int)
    parser.add_argument("-nc", "--no-clock", help="Do not inset clock", action='store_true')
    parser.add_argument("-fn", "--first_n", help="Process only the first N frames", type=int)
    parser.add_argument("-i", "--interval", help="Percentile to use for scaling", default=99.9, type=float)
    args = parser.parse_args()

    if args.selektor:
        client = EUISelektorClient()
        query = parse_query_args(args.selektor)
        res = client.search_nolimit(query)
        if res is not None:
            archive_path = os.getenv('EUI_ARCHIVE_DATA_PATH')
            if archive_path == '':
                print('Warning: undefined EUI_ARCHIVE_DATA_PATH')
            files = [eui_file2path(os.path.basename(f), archive_path) for f in res['filepath']]
            args.source = [f for f in files if os.path.exists(f)]
            if len(args.source) == 0:
                sys.exit(1)
        else:
            sys.exit(1)

    main(**vars(args))
