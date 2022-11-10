import os
import glob
from .image import Sequence
from astropy.io import fits
import matplotlib as mp

mp.use('Agg')


def main(**kwargs):
    source = kwargs['source']
    if type(source) is not list:
        source = [source]
    files = []
    for s in source:
        if os.path.isfile(s):
            files.append(s)
        else:
            if os.path.isdir(s):
                s = os.path.join(s, '*.fits')
            files.append(glob.glob(s))
    if len(files) == 0:
        print('No files found')
        return
    files.sort()
    if 'first_n' in kwargs:
        files = files[0:kwargs['first_n']]
    seq = Sequence(files, **kwargs)
    seq.process()
