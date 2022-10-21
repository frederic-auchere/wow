import argparse
import os
import glob
import cv2
import numpy as np
import json
from image import Image
from generic import make_directory
from tqdm import tqdm
from scipy.ndimage import correlate

parser = argparse.ArgumentParser()
parser.add_argument("source", help="List of files", type=str)
parser.add_argument("-o", "--output_directory", help="Output directory", default=None, type=str)
parser.add_argument("-r", "--reference", help="Reference image number", default=-1, type=int)


def cross_correlate(image, reference):
    image = np.copy(np.sqrt(np.array(image)[512:512+1024, 512:512+1024]))
    cc = cv2.matchTemplate(np.sqrt(np.array(reference)), image, cv2.TM_CCOEFF_NORMED)
    cy, cx = np.unravel_index(np.argmax(cc, axis=None), cc.shape)

    xi = [cx - 1, cx, cx + 1]
    yi = [cy - 1, cy, cy + 1]
    ccx2 = cc[[cy, cy, cy], xi]**2
    ccy2 = cc[yi, [cx, cx, cx]]**2

    xn = ccx2[2] - ccx2[1]
    xd = ccx2[0] - 2*ccx2[1] + ccx2[2]
    yn = ccy2[2] - ccy2[1]
    yd = ccy2[0] - 2*ccy2[1] + ccy2[2]

    if xd != 0:
        dx = xi[2] - (xn / xd + 0.5)
    else:
        dx = cx
    if yd != 0:
        dy = yi[2] - (yn / yd + 0.5)
    else:
        dy = cy

    dx -= 512
    dy -= 512

    return dx, dy


def process(source, **kwargs):
    files = glob.glob(source)
    if len(files) == 0:
        print('No files found')
        return
    files.sort()
    if kwargs['reference'] == -1:
        kwargs['reference'] = len(files)//2
    elif kwargs['reference'] > len(files):
        kwargs['reference'] = 0
    reference = Image(files[kwargs['reference']])
    reference_crvals = reference.header['CRVAL1'], reference.header['CRVAL2']
    output = {}
    for f in tqdm(files, desc='Registering'):
        image = Image(f)
        dx, dy = cross_correlate(image, reference) if f != files[kwargs['reference']] else (0, 0)
        dx *= image.header['CDELT1']
        dy *= image.header['CDELT2']
        crota = np.radians(image.header['CROTA'])
        delta_crval1 = np.cos(-crota) * dx - np.sin(-crota) * dy
        delta_crval2 = np.sin(-crota) * dx + np.cos(-crota) * dy
        output[f] = image.header['CRVAL1'] - reference_crvals[0] - delta_crval1, \
                    image.header['CRVAL2'] - reference_crvals[1] - delta_crval2
    output_directory = make_directory(kwargs['output_directory'])
    with open(os.path.join(output_directory, 'register.json'), 'w') as f:
        json.dump(output, f)


def main(**kwargs):
    source = kwargs['source']
    if os.path.isdir(source):
        kwargs['source'] = os.path.join(kwargs['source'], '*.fits')
    process(**kwargs)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
