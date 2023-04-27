import os
import cv2
import numpy as np
from astropy.io import fits
from watroo import utils
from rectify.rectify import HomographicTransform, Rectifier, rotationmatrix
from astropy.visualization import ImageNormalize, LinearStretch, AsymmetricPercentileInterval
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib as mp
from multiprocessing import Pool, cpu_count
from tempfile import NamedTemporaryFile
from tqdm import tqdm
import subprocess
from .generic import make_directory, rebin
from .plotting import make_subplot
from sunpy.visualization.colormaps import cm


def fsi_fix_shift(img):
    a = np.median(img, axis=0)
    if a[0] > 2 * np.mean(a[1:4]):
        img[:, 0:-1] = img[:, 1:]
        img[:, -1] = 0


def read(source):
    ext = os.path.splitext(source)[1]
    file = os.path.split(source)[1]
    if ext == '.fit' or ext == '.fits' or ext == '.fts' or 'efz' in file:
        with fits.open(source) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    image = np.float32(hdu.data)
                    header = hdu.header
                    break
        if 'eui-fsi' in source:
            fsi_fix_shift(image)
            if '_L1_' in header['FILENAME']:
                image /= header['XPOSURE']
            if header['WAVELNTH'] == 304:
                dn_per_photon = 3.8803
            else:
                dn_per_photon = 7.3776
            gain = dn_per_photon
            read_noise = 1.5
        elif 'eui-hrieuv' in source:
            if '_L1_' in source:
                image /= header['XPOSURE']
            dn_per_photon = 5.2764
            gain = dn_per_photon
            read_noise = 1.5
        elif 'aia' in source:
            header['XPOSURE'] = header['EXPTIME']
            image /= header['XPOSURE']
            electron_per_dn = 17.7
            electron_per_photon = 13.6 * 911.0 / (3.65 * 171)
            dn_per_photon = electron_per_photon / electron_per_dn
            gain = dn_per_photon
            read_noise = 1.15
        elif 'lasco' in source:
            header['XPOSURE'] = header['EXPTIME']
            image /= header['XPOSURE']
            electron_per_photon = 1
            electron_per_dn = 15
            dn_per_photon = electron_per_photon / electron_per_dn
            gain = dn_per_photon
            read_noise = 0.3
        elif 'metis' in source:
            if 'XPOSURE' in header:
                image /= header['XPOSURE']
            read_noise = None
            gain = None
            dn_per_photon = 1
        elif 'efz' in file:
            if 'EXPTIME' in header:
                image /= header['EXPTIME']
            header['DATE-OBS'] = header['DATE_OBS']
            read_noise = None
            gain = None
            dn_per_photon = 1
        else:
            read_noise = None
            gain = None
            dn_per_photon = 1
        image /= dn_per_photon
    else:
        image = cv2.imread(source)
        image = np.float32(image.sum(axis=2))
        image = np.flipud(image)
        header = None

    return image, header, gain, read_noise


class Image:

    cmaps = {'Unknwon': 'gray',
             'eit171': 'sohoeit171',
             'eit195': 'sohoeit195',
             'eit284': 'sohoeit284',
             'eit304': 'sohoeit304',
             'hrieuv': 'solar orbiterhri_euv174',
             'hrilya': 'solar orbiterhri_lya1216',
             'fsi174': 'solar orbiterfsi174',
             'fsi304': 'solar orbiterfsi304',
             'aia94': 'sdoaia94',
             'aia131': 'sdoaia131',
             'aia171': 'sdoaia171',
             'aia193': 'sdoaia193',
             'aia1211': 'sdoaia211',
             'aia304': 'sdoaia304',
             'aia335': 'sdoaia335',
             'metis': 'gray'
             }

    def __init__(self, source, roi=None):
        self.source = source
        self._data = None
        self._noise = None
        self._header = None
        if roi is not None:
            self.roi = slice(roi[1], roi[3]), slice(roi[0], roi[2])
        else:
            self.roi = None
        self._instrument = None
        self._cmap = None

    def __array__(self):
        return self.data

    @property
    def instrument(self):
        if self._instrument is None:
            if 'TELESCOP' in self.header:
                if self.header['TELESCOP'] == 'SOLO/EUI/HRI_EUV':
                    self._instrument = 'hrieuv'
                elif self.header['TELESCOP'] == 'SOLO/EUI/HRI_LYA':
                    self._instrument = 'hrilya'
                elif self.header['TELESCOP'] == 'SOLO/EUI/FSI':
                    self._instrument = 'fsi' + str(self.header['WAVELNTH'])
                elif self.header['TELESCOP'] == 'SDO/AIA':
                    self._instrument = 'aia' + str(self.header['WAVELNTH'])
                elif 'Metis' in self.header['TELESCOP']:
                    self._instrument = 'metis'
                elif self.header['TELESCOP'] == 'SOHO':
                    if self.header['INSTRUME'] == 'EIT':
                        self._instrument = 'eit' + str(self.header['WAVELNTH'])
                    else:
                        self._instrument = None
                else:
                    self._instrument = None
            else:
                self._instrument = None
        return self._instrument

    @property
    def cmap(self):
        if self._cmap is None:
            self._cmap = plt.get_cmap(self.cmaps[self.instrument])
        return self._cmap

    @property
    def data(self):
        if self._data is None:
            self.read()
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def header(self):
        if self._header is None:
            with fits.open(self.source) as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        self._header = hdu.header
                        break
        return self._header

    @header.setter
    def header(self, header):
        self._header = header

    @property
    def noise(self):
        if self._noise is None\
         and self.header['GAIN'] is not None\
         and self.header['RDNOISE'] is not None\
         and 'XPOSURE' in self.header:
            image = self.data*self.header['XPOSURE']
            image[image < 0] = 0
            self._noise = np.sqrt(self.header['GAIN'] * image + self.header['RDNOISE'] ** 2)/self.header['XPOSURE']
            self._noise[self._noise <= 0] = 1
        return self._noise

    def read(self, array=None):
        image, header, gain, read_noise = read(self.source)
        header['GAIN'] = gain
        header['RDNOISE'] = read_noise
        if self._data is None:
            if self.roi is None:
                self.roi = slice(0, image.shape[1]), slice(0, image.shape[0])
        header['CRPIX1'] -= self.roi[1].start
        header['CRPIX2'] -= self.roi[0].start
        image = image[self.roi]
        if array is None:
            array = image
        else:
            array[:] = image
        self._data = array
        self.header = header

    def geometric_rectification(self, target=None, north_up=True, center=True, order=2, opencv=True):

        if target is None:
            if center:
                target = (0, 0)
            else:
                target = self.header['CRVAL1'], self.header['CRVAL2']

        alpha1 = -np.radians(self.header['CRVAL1']/3600)
        beta1 = np.radians(self.header['CRVAL2']/3600)
        alpha2 = np.radians(target[0]/3600)
        beta2 = -np.radians(target[1]/3600)

        # gamma = -np.radians(self.header['CROTA']) if north_up else 0
        gamma1 = -np.radians(self.header['CROTA'])
        gamma2 = -gamma1 if not north_up else 0

        # x, y in the plane of the sky, z towards the observer
        # yaw = around y, pitch = around x, roll = around z
        Ry1 = rotationmatrix(alpha1, 1)
        Rx1 = rotationmatrix(beta1, 2)
        Rz1 = rotationmatrix(gamma1, 0)
        Ry2 = rotationmatrix(alpha2, 1)
        Rx2 = rotationmatrix(beta2, 2)
        Rz2 = rotationmatrix(gamma2, 0)

        # K & K^{-1}
        Km1 = np.array([[np.radians(self.header['CDELT1']/3600), 0, -(self.header['CRPIX1']-1)*np.radians(self.header['CDELT1']/3600)],
                        [0, np.radians(self.header['CDELT2']/3600), -(self.header['CRPIX2']-1)*np.radians(self.header['CDELT2']/3600)],
                        [0, 0, 1]])
        K = np.array([[1/np.radians(self.header['CDELT1']/3600), 0, self.header['CRPIX1']-1],
                      [0, 1/np.radians(self.header['CDELT2']/3600), self.header['CRPIX2']-1],
                      [0, 0, 1]])
        KH = K @ Rz1 @ Rx1 @ Ry1 @ Ry2 @ Rx2 @ Rz2 @ Km1
        KH /= KH[2, 2]

        transform = HomographicTransform(KH, dtype=np.float32)

        rectifier = Rectifier(transform)

        xfov = 0, self.data.shape[1]-1
        yfov = 0, self.data.shape[0]-1

        self.data[:] = rectifier(self.data, self.data.shape, xfov, yfov, order=order, opencv=opencv)

        if center:
            self.header['CRVAL1'] = 0
            self.header['CRVAL2'] = 0
        if north_up:
            self.header['PC1_1'] = 1
            self.header['PC1_2'] = 0
            self.header['PC2_1'] = 0
            self.header['PC2_2'] = 1
            self.header['CROTA'] = 0

    def rebin(self, binning):
        self.header['NAXIS1'] /= binning
        self.header['NAXIS2'] /= binning
        self.header['CDELT1'] *= binning
        self.header['CDELT2'] *= binning
        self.header['CRPIX1'] = (self.header['NAXIS1'] + 1) / 2
        self.header['CRPIX2'] = (self.header['NAXIS2'] + 1) / 2
        self.data = rebin(self.data, [s // binning for s in self.data.shape])


class Sequence:
    def __init__(self, files, **kwargs):
        self.frames = [Image(f, roi=kwargs['roi']) for f in files]
        self.kwargs = kwargs
        output_directory, output_file =\
            os.path.split(self.kwargs['output']) if kwargs['output'] is not None else ('', '')
        self.output_directory = make_directory(output_directory)
        kwargs['output_directory'] = self.output_directory
        self.output_file = 'wow.mp4' if output_file == '' else output_file
        self.xy = self.tracking(order=kwargs['register']) if kwargs['register'] >= 0 else (None,)*len(self.frames)

    def tracking(self, order=2):
        crval1 = [f.header['CRVAL1'] for f in self.frames]
        crval2 = [f.header['CRVAL2'] for f in self.frames]
        date = [f.header['DATE-OBS'] for f in self.frames]

        time = Time(date).mjd

        polynomial = np.polynomial.Polynomial
        crval1_fit = polynomial.fit(time, crval1, order)
        crval2_fit = polynomial.fit(time, crval2, order)

        return [(crval1_fit(t), crval2_fit(t)) for t in time]

    @property
    def _is_consistent(self, keys=('NAXIS1', 'NAXIS2')):
        return all(all(f.header[k] == self.frames[0].header[k] for k in keys) for f in self.frames[1:])

    def process(self):
        def from_header(frame):
            is_fsi = 'FSI' in frame.header['TELESCOP'] if 'TELESCOP' in frame.header else False
            return {
                    'is_fsi': is_fsi,
                    }

        fps = self.kwargs["frame_rate"]
        writer = NamedTemporaryFile(delete=False)
        gamma_min, gamma_max = AsymmetricPercentileInterval(*self.kwargs['interval']).get_limits(self.frames[0].data)

        if self.kwargs['temporal']:
            cube = self.prep_cube(gamma_min=gamma_min, gamma_max=gamma_max)
            norm = ImageNormalize(cube, interval=AsymmetricPercentileInterval(*self.kwargs['interval']), stretch=LinearStretch())
        else:
            norm, _ = self.process_single_frame({**self.kwargs,
                                                 **{'source': self.frames[0].source,
                                                    'gamma_min': gamma_min,
                                                    'gamma_max': gamma_max,
                                                    'register': False,
                                                    'enhance': True},
                                                 **from_header(self.frames[0])}
                                                )
        pool_args = [{**self.kwargs,
                      **{'source': f.source,
                         'gamma_min': gamma_min,
                         'gamma_max': gamma_max,
                         'data': np.copy(f.data) if self.kwargs['temporal'] else None,
                         'xy': xy,
                         'norm': norm,
                         'register': self.kwargs['register'] >= 0 and not self.kwargs['temporal'],
                         'enhance': not self.kwargs['temporal']},
                      **from_header(f)
                      } for f, xy in zip(self.frames, self.xy)]
        with Pool(max(1, cpu_count()-1) if self.kwargs['n_procs'] == 0 else self.kwargs['n_procs']) as pool:
            res = list(tqdm(pool.imap(self.process_single_frame, pool_args), desc='Processing', total=len(self.frames)))
        for _, file_name in res:
            line = "file '" + os.path.abspath(file_name) + "'\n"
            writer.write(line.encode())
            line = f"duration {1 / fps:.2f}\n"
            writer.write(line.encode())
        writer.close()
        if not self.kwargs['no_encode'] and len(self.frames) > 1:
            crf = self.kwargs["crf"]
            subprocess.run(["ffmpeg",
                            "-f", "concat",
                            "-safe", "0",
                            "-i", writer.name,
                            "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-crf", f"{crf}",
                            "-r", f"{fps}",
                            "-y", os.path.join(self.output_directory, self.output_file)])
        os.unlink(writer.name)

    @staticmethod
    def process_single_frame(kwargs):

        def make_frame(image, title=None, norm=None, clock=None, cmap='gray'):
            dpi = 300
            fig_size = [s / dpi for s in image.shape[::-1]]
            fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
            mp.rc('font', size=12 * dpi * fig_size[1] / 2048)

            if norm is None:
                norm = ImageNormalize(image, interval=AsymmetricPercentileInterval(0.1, 99.9), stretch=LinearStretch())
            make_subplot(image, ax, norm, cmap=cmap, title=title, clock=clock)
            fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

            return fig, ax

        source = kwargs['source']
        image = Image(source, roi=kwargs['roi'])
        if 'data' in kwargs:
            if kwargs['data'] is not None:
                image.data = kwargs['data']

        if kwargs['enhance']:
            gamma_min = kwargs['gamma_min'] if 'gamma_min' in kwargs else None
            gamma_max = kwargs['gamma_max'] if 'gamma_max' in kwargs else None
            image.data, _ = utils.wow(image.data,
                                      denoise_coefficients=kwargs['denoise'],
                                      noise=image.noise,
                                      n_scales=kwargs['n_scales'],
                                      bilateral=None if kwargs['no_bilateral'] else 1,
                                      whitening=not kwargs['no_whitening'],
                                      gamma=kwargs['gamma'],
                                      h=kwargs['gamma_weight'],
                                      gamma_min=gamma_min,
                                      gamma_max=gamma_max)

        if kwargs['register']:
            is_fsi = kwargs['is_fsi'] if 'is_fsi' in kwargs else False
            xy = kwargs['xy'] if 'xy' in kwargs else None  # and not is_fsi else None
            image.geometric_rectification(target=xy, north_up=is_fsi, center=is_fsi)

        clock = None if kwargs['no_clock'] else image.header['DATE-OBS']
        if 'norm' in kwargs:
            norm = kwargs['norm']
        else:
            mini, maxi = AsymmetricPercentileInterval(*kwargs['interval']).get_limits(image.data)
            norm = ImageNormalize(vmin=mini, vmax=maxi, stretch=LinearStretch())
        label = image.header['DATE-OBS'][:-4]
        if kwargs['gamma_weight'] < 1:
            label += ' WOW-enhanced'
        if kwargs['rebin'] > 1:
            image.rebin(kwargs['rebin'])
        fig, ax = make_frame(image.data, title=label, norm=norm, clock=clock, cmap=image.cmap)

        output_directory = kwargs['output_directory']
        out_file = os.path.join(output_directory, os.path.basename(image.source))

        try:
            if kwargs['to_fits']:
                image.header['COMMENT'] = '--------- WOW parameters: ------------------------------------'
                image.header['GAMMA'] = kwargs['gamma']
                image.header['GAMMAW'] = kwargs['gamma_weight']
                image.header['DENOISE'] = str(kwargs['denoise'])
                image.header['BILATERA'] = 'None' if kwargs['no_bilateral'] else 1
                image.header['TEMPORAL'] = 'True' if kwargs['temporal'] else 'False'
                image.header['INTERVAL'] = str(kwargs['interval'])
                fits.writeto(out_file, image.data, header=image.header, overwrite=True)
            out_file += '.png'
            fig.savefig(out_file)
            plt.close(fig)
        except IOError:
            raise IOError

        return norm, out_file

    def prep_cube(self, gamma_min=None, gamma_max=None):
        for i, (f, xy) in tqdm(enumerate(zip(self.frames, self.xy)), desc='Reading files', total=len(self.frames)):
            if i == 0:
                cube = np.empty(shape=((len(self.frames),) + f.data.shape), dtype=np.float32)
                if f.noise is not None:
                    noise = np.empty(shape=((len(self.frames),) + f.data.shape), dtype=np.float32)
                else:
                    noise = None
                is_fsi = 'FSI' in f.header['TELESCOP'] if 'TELESCOP' in f.header else False

            f.read(array=cube[i])
            if self.kwargs['register'] >= 0:
                f.geometric_rectification(target=xy, north_up=is_fsi, center=is_fsi)

            if f.noise is not None:
                noise[i] = f.noise

        cube[:], _ = utils.wow(cube,
                               denoise_coefficients=self.kwargs['denoise'],
                               noise=noise,
                               n_scales=self.kwargs['n_scales'],
                               bilateral=None if self.kwargs['no_bilateral'] else 1,
                               whitening=not self.kwargs['no_whitening'],
                               gamma=self.kwargs['gamma'],
                               h=self.kwargs['gamma_weight'],
                               gamma_min=gamma_min,
                               gamma_max=gamma_max)
        return cube
