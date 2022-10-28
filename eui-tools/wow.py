import argparse
import os
import glob
import json
from watroo import utils
from image import Sequence
from plotting import make_subplot
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, PercentileInterval, LinearStretch
from tqdm import tqdm
import subprocess
import numpy as np
from tempfile import NamedTemporaryFile
from generic import make_directory
from sunpy.visualization.colormaps import cm
from multiprocessing import Pool, cpu_count
import matplotlib as mp
from astropy.io import fits

mp.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("source", help="List of files", type=str)
parser.add_argument("-o", "--output_directory", help="Output directory", default=None, type=str)
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
parser.add_argument("-ck", "--clock", help="Inset clock", action='store_true')
parser.add_argument("-fn", "--first_n", help="Process only the first N frames", type=int)
parser.add_argument("-i", "--interval", help="Percentile to use for scaling", default=99.9, type=float)
parser.add_argument("-ex", "--exposure", help="Minimum exposure time", type=float)


def make_frame(image, title=None, norm=None, clock=None):
    dpi = 300
    fig_size = [s / dpi for s in image.data.shape[::-1]]
    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
    mp.rc('font', size=12 * dpi * fig_size[1] / 2048)

    if norm is None:
        norm = ImageNormalize(image, interval=PercentileInterval(99.9), stretch=LinearStretch())
    cmap = plt.get_cmap('solar orbiterhri_euv174')
    make_subplot(image, ax, norm, cmap=cmap, title=title, clock=clock)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    return fig, ax


def process_single_file(kwargs):
    source = kwargs['source']
    xy = kwargs['xy'] if 'xy' in kwargs else None
    image = Image(source, roi=kwargs['roi'])

    gamma_min, gamma_max = image.enhance(kwargs)

    if kwargs['register']:
        is_fsi = 'FSI' in image.header['TELESCOP'] if 'TELESCOP' in image.header else False
        image.geometric_rectification(target=xy, north_up=is_fsi, center=is_fsi)

    clock = image.header['DATE-OBS'] if 'clock' in kwargs else None
    if 'norm' in kwargs:
        norm = kwargs['norm']
    else:
        norm = ImageNormalize(image.data, interval=PercentileInterval(kwargs['interval']), stretch=LinearStretch())
    fig, ax = make_frame(image.data, title=image.header['DATE-OBS'][:-4], norm=norm, clock=clock)
    norm = ax.get_images()[0].norm

    output_directory = make_directory(kwargs['output_directory'])
    out_file = os.path.join(output_directory, os.path.basename(source + '.png'))

    try:
        fig.savefig(out_file)
        plt.close(fig)
    except IOError:
        raise IOError

    xy = image.header["CRVAL1"], image.header["CRVAL2"]
    return norm, gamma_min, gamma_max, xy, out_file


def process(source, **kwargs):
    if os.path.isfile(source) and os.path.splitext(source)[1] == '.json':
        with open(source) as json_file:
            records = json.load(json_file)
        files = list(records.keys())
        dxy = list(records.values())
    else:
        files = glob.glob(source)
        if len(files) == 0:
            print('No files found')
            return
        if 'exposure' in kwargs:
            files = [f for f in files if fits.getheader(f, 1)['XPOSURE'] > kwargs['exposure']]
        files.sort()
        dxy = ((0, 0),) * len(files)
    if 'first_n' in kwargs:
        files = files[0:kwargs['first_n']]
    fps = kwargs["frame_rate"]
    writer = NamedTemporaryFile(delete=False)
    output_directory = make_directory(kwargs['output_directory'])
    if not kwargs['temporal']:
        norm, gamma_min, gamma_max, xy, _ = process_single_file({**{'source': files[0], 'interval': kwargs['interval']},
                                                                 **kwargs})
        nxy = [(xy[0] + dx, xy[1] + dy) for (dx, dy) in dxy]
        pool_args = [{**{'source': f, 'xy': n, 'norm': norm, 'gamma_min': gamma_min, 'gamma_max': gamma_max},
                      **kwargs} for f, n in zip(files, nxy)]
        with Pool(cpu_count() if kwargs['n_procs'] == 0 else kwargs['n_procs']) as pool:
            res = list(tqdm(pool.imap(process_single_file, pool_args), desc='Processing', total=len(files)))
        for _, _, _, _, file_name in res:
            line = "file '" + os.path.abspath(file_name) + "'\n"
            writer.write(line.encode())
            line = f"duration {1/fps:.2f}\n"
            writer.write(line.encode())
    else:
        for i, f in tqdm(enumerate(files), desc='Reading files'):
            image = Image(f, roi=kwargs['roi'])
            image.data[image.data < 0] = 0
            if i == 0:
                cube = np.empty(shape=image.shape + (len(files),))
                noise = np.empty(shape=image.shape + (len(files),))
                x0 = image.header["CRVAL1"] / image.header["CDELT1"]
                y0 = image.header["CRVAL2"] / image.header["CDELT2"]
            else:
                if kwargs['register']:
                    image.register(image, x0=x0, y0=y0)
            cube[:, :, i] = image
            noise[:, :, i] = image.noise
        out, _ = utils.wow(cube,
                           denoise_coefficients=kwargs['denoise'],
                           noise=noise,
                           n_scales=kwargs['n_scales'],
                           bilateral=None if kwargs['no_bilateral'] else 1,
                           whitening=not kwargs['no_whitening'],
                           gamma=kwargs['gamma'],
                           h=kwargs['gamma_weight'])
        norm = ImageNormalize(out, interval=PercentileInterval(kwargs['interval']), stretch=LinearStretch())
        for i, f in tqdm(enumerate(files), desc='Writing files'):
            fig, ax = make_frame(out[:, :, i], title=None, norm=norm)
            out_file = os.path.join(output_directory, os.path.basename(f + '.png'))
            try:
                fig.savefig(out_file)
                plt.close(fig)
            except IOError:
                raise IOError
            line = "file '" + os.path.abspath(out_file) + "'\n"
            writer.write(line.encode())
            line = f"duration {1/fps:.2f}\n"
            writer.write(line.encode())
    writer.close()
    if not kwargs['no_encode'] and len(files) > 1:
        subprocess.run(["ffmpeg",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", writer.name,
                        "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", "22",
                        "-r", f"{fps}",
                        "-y", os.path.join(output_directory, 'wow.mp4')])
    os.unlink(writer.name)


def main(**kwargs):
    source = kwargs['source']
    if os.path.isdir(source):
        source = os.path.join(kwargs['source'], '*.fits')
    files = glob.glob(source)
    if len(files) == 0:
        print('No files found')
        return
    files.sort()
    if 'first_n' in kwargs:
        files = files[0:kwargs['first_n']]
    seq = Sequence(files, **kwargs)
    seq.process()


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
