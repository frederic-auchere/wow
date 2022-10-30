import argparse
import os
import glob
from .image import Sequence
from astropy.io import fits
import matplotlib as mp

mp.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("source", help="List of files", type=str)
parser.add_argument("-o", "--output_directory", help="Output directory", default='.', type=str)
parser.add_argument("-d", "--denoise", help="Denoising coefficients", default=[], type=float, nargs='+')
parser.add_argument("-nb", "--no_bilateral", help="Do not use edge-aware (bilateral) transform", action='store_true')
parser.add_argument("-ns", "--n_scales", help="Number of wavelet scales", default=None, type=int)
parser.add_argument("-gw", "--gamma_weight", help="Weight of gamma-stretched image", default=0, type=float)
parser.add_argument("-g", "--gamma", help="Gamma exponent", default=2, type=float)
parser.add_argument("-nw", "--no_whitening", help="Do not apply whitening (WOW!)", action='store_true')
parser.add_argument("-t", "--temporal", help="Applies temporal denoising and/or whitening", action='store_true')
parser.add_argument("-roi", help="Region of interest [bottom left, top right corners]", type=int, nargs=4)
parser.add_argument("-r", "--register", help="Uses header information to register the frames", action='store_true')
parser.add_argument("-ne", "--no_encode", help="Do not encode the frames to video", action='store_true')
parser.add_argument("-fps", "--frame-rate", help="Number of frames per second", default=12, type=float)
parser.add_argument("-np", "--n_procs", help="Number of processors to use", default=0, type=int)
parser.add_argument("-nc", "--no-clock", help="Inset clock", action='store_true')
parser.add_argument("-fn", "--first_n", help="Process only the first N frames", type=int)
parser.add_argument("-i", "--interval", help="Percentile to use for scaling", default=99.9, type=float)
parser.add_argument("-ex", "--exposure", help="Minimum exposure time", type=float)


def main(**kwargs):
    source = kwargs['source']
    if os.path.isdir(source):
        source = os.path.join(kwargs['source'], '*.fits')
    files = glob.glob(source)
    if len(files) == 0:
        print('No files found')
        return
    files.sort()
    if 'exposure' in kwargs:
        files = [f for f in files if fits.getheader(f, 1)['XPOSURE'] > kwargs['exposure']]
    if 'first_n' in kwargs:
        files = files[0:kwargs['first_n']]
    seq = Sequence(files, **kwargs)
    seq.process()


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
