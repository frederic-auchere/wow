import os
import glob
from .image import Sequence
from astropy.io import fits
import matplotlib as mp

mp.use('Agg')


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
        if kwargs['exposure'] is not None:
            files = [f for f in files if fits.getheader(f, 1)['XPOSURE'] > kwargs['exposure']]
    if 'first_n' in kwargs:
        files = files[0:kwargs['first_n']]
    seq = Sequence(files, **kwargs)
    seq.process()
