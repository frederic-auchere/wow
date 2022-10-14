import argparse
import os
import glob
from watroo import utils
from generic import read_data, register, make_subplot, data_noise
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, PercentileInterval, LinearStretch
from tqdm import tqdm
import subprocess
import numpy as np
from tempfile import NamedTemporaryFile
from sunpy.visualization.colormaps import cm
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser()
parser.add_argument("source", help="List of files", type=str)
parser.add_argument("-o", "--output_directory", help="Output directory", default=None, type=str)
parser.add_argument("-d", "--denoise", help="Denoising coefficients", default=[], type=float, nargs="+")
parser.add_argument("-nb", "--no_bilateral", help="Do not use edge-aware (bilateral) transform", action="store_true")
parser.add_argument("-ns", "--n_scales", help="Number of wavelet scales", default=None, type=int)
parser.add_argument("-gw", "--gamma_weight", help="Weight of gamma-stretched image", default=0, type=float)
parser.add_argument("-g", "--gamma", help="Gamma exponent", default=2, type=float)
parser.add_argument("-nw", "--no_whitening", help="Do not apply whitening (WOW!)", action="store_true")
parser.add_argument("-bf", "--by_frame", help="Applies denoising and/or whitening by frame", action="store_true")
parser.add_argument("-roi", help="Region of interest [bottom left, top right corners]", type=int, nargs=4)
parser.add_argument("-f", "--flicker", help="Uses different normalization for each frame", action="store_true")
parser.add_argument("-r", "--register", help="Uses header information to register the frames", action="store_true")
parser.add_argument("-ne", "--no_encode", help="Do not encode the frames to video", action="store_true")
parser.add_argument("-fps", "--frame-rate", help="Number of frames per second", default=12, type=float)
parser.add_argument("-np", "--n_procs", help="Number of processors to use", default=0, type=int)


def make_directory(name):
    if name:
        if os.path.isdir(name):
            return name
        try:
            os.mkdir(name)
            return name
        except FileNotFoundError:
            return ''
    else:
        return ''


def make_frame(image, title=None, norm=None):
    dpi = 300
    fig_size = [s/dpi for s in image.shape]
    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)

    if norm is None:
        norm = ImageNormalize(image, interval=PercentileInterval(99.9), stretch=LinearStretch())
    cmap = plt.get_cmap('solar orbiterhri_euv174')
    make_subplot(image, ax, norm, cmap=cmap, title=title)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    return fig, ax


def process_single_file(kwargs):
    source = kwargs['source']
    norm = kwargs['norm'] if 'norm' in kwargs else None
    gamma_min = kwargs['gamma_min'] if 'gamma_min' in kwargs else None
    gamma_max = kwargs['gamma_max'] if 'gamma_max' in kwargs else None
    data = {'file': source, 'roi': kwargs['roi']}
    image, header = read_data(data)
    noise = data_noise(image, data)

    if gamma_min is None:
        gamma_min = image.min()
    if gamma_max is None:
        gamma_max = image.max()

    out, _ = utils.wow(image,
                       denoise_coefficients=kwargs['denoise'],
                       noise=noise,
                       n_scales=kwargs['n_scales'],
                       bilateral=None if kwargs['no_bilateral'] else 1,
                       whitening=not kwargs['no_whitening'],
                       gamma=kwargs['gamma'],
                       h=kwargs['gamma_weight'],
                       gamma_min=gamma_min,
                       gamma_max=gamma_max)

    fig, ax = make_frame(out, title=header['DATE-OBS'][:-4], norm=norm)
    norm = ax.get_images()[0].norm

    output_directory = make_directory(kwargs['output_directory'])
    out_file = os.path.join(output_directory, os.path.basename(source + '.png'))

    try:
        fig.savefig(out_file)
        plt.close(fig)
    except IOError:
        raise IOError

    return norm, gamma_min, gamma_max, out_file


def process(source, **kwargs):
    files = glob.glob(source)
    files.sort()
    fps = kwargs["frame_rate"]
    writer = NamedTemporaryFile(delete=False)
    output_directory = make_directory(kwargs['output_directory'])
    if kwargs['by_frame']:
        norm, gamma_min, gamma_max, _ = process_single_file({**{'source': files[0]}, **kwargs})
        if kwargs['flicker']:
            norm, gamma_min, gamma_max = None, None, None
        with Pool(cpu_count() if kwargs['n_procs'] == 0 else kwargs['n_procs']) as pool:
            args = [{**{'source': f, 'norm': norm, 'gamma_min': gamma_min, 'gamma_max': gamma_max}, **kwargs} for f in files]
            res = list(tqdm(pool.imap(process_single_file, args), desc='Processing'))
            for _, _, _, file_name in res:
                line = "file '" + os.path.abspath(file_name) + "'\n"
                writer.write(line.encode())
                line = f"duration {1/fps:.2f}\n"
                writer.write(line.encode())
    else:
        for i, f in tqdm(enumerate(files), desc='Reading files'):
            data = {'file': f, 'roi': kwargs['roi']}
            image, header = read_data(data)
            if i == 0:
                cube = np.empty(shape=image.shape + (len(files),))
            image[image < 0] = 0
            cube[:, :, i] = image
        noise = data_noise(cube, data)
        out, _ = utils.wow(cube,
                           denoise_coefficients=kwargs['denoise'],
                           noise=noise,
                           n_scales=kwargs['n_scales'],
                           bilateral=None if kwargs['no_bilateral'] else 1,
                           whitening=not kwargs['no_whitening'],
                           gamma=kwargs['gamma'],
                           h=kwargs['gamma_weight'])
        norm = ImageNormalize(out, interval=PercentileInterval(99.9), stretch=LinearStretch())
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
                        "-crf", "20",
                        "-r", f"{fps}",
                        "-y", os.path.join(output_directory, 'wow.mp4')])
    os.unlink(writer.name)


def main(**kwargs):
    source = kwargs['source']
    if os.path.isdir(source):
        kwargs['source'] = os.path.join(kwargs['source'], '*.fits')
    process(**kwargs)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
