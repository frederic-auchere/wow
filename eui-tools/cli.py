import argparse
from wow import main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="List of files", type=str)
    parser.add_argument("-o", "--output_directory", help="Output directory", default=None, type=str)
    parser.add_argument("-d", "--denoise", help="Denoising coefficients", default=[], type=float, nargs='+')
    parser.add_argument("-nb", "--no_bilateral", help="Don't use edge-aware (bilateral) transform", action='store_true')
    parser.add_argument("-ns", "--n_scales", help="Number of wavelet scales", default=None, type=int)
    parser.add_argument("-gw", "--gamma_weight", help="Weight of gamma-stretched image", default=0, type=float)
    parser.add_argument("-g", "--gamma", help="Gamma exponent", default=2, type=float)
    parser.add_argument("-nw", "--no_whitening", help="Do not apply whitening (WOW!)", action='store_true')
    parser.add_argument("-t", "--temporal", help="Applies temporal denoising and/or whitening", action='store_true')
    parser.add_argument("-roi", help="Region of interest [bottom left, top right corners]", type=int, nargs=4)
    parser.add_argument("-f", "--flicker", help="Uses different normalization for each frame", action='store_true')
    parser.add_argument("-r", "--register", help="Uses header information to register the frames", action='store_true')
    parser.add_argument("-ne", "--no_encode", help="Do not encode the frames to video", action='store_true')
    parser.add_argument("-fps", "--frame-rate", help="Number of frames per second", default=12, type=float)
    parser.add_argument("-np", "--n_procs", help="Number of processors to use", default=0, type=int)
    parser.add_argument("-ck", "--clock", help="Inset clock", action='store_true')
    parser.add_argument("-fn", "--first_n", help="Process only the first N frames", type=int)
    parser.add_argument("-i", "--interval", help="Percentile to use for scaling", default=99.9, type=float)

    args = parser.parse_args()
    main(**vars(args))
