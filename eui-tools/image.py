import os
import cv2
import numpy as np
from astropy.io import fits
from rectify.rectify import EuclidianTransform, HomographicTransform, Rectifier, rotationmatrix


def read(filename):
    ext = os.path.splitext(filename)[1]
    if ext == '.fit' or ext == '.fits' or ext == '.fts':
        with fits.open(filename) as hdul:
            image = np.float32(hdul[-1].data)
            header = hdul[-1].header
        if 'eui-fsi' in filename:
            if '_L2_' in filename:
                image *= header['XPOSURE']
            if '304' in filename:
                dn_per_photon = 3.8803
            else:
                dn_per_photon = 7.3776
            gain = dn_per_photon
            read_noise = 1.5
        elif 'eui-hrieuv' in filename:
            if '_L2_' in filename:
                image *= header['XPOSURE']
            dn_per_photon = 5.2764
            gain = dn_per_photon
            read_noise = 1.5
        elif 'aia' in filename:
            electron_per_dn = 17.7
            electron_per_photon = 13.6 * 911.0 / (3.65 * 171)
            dn_per_photon = electron_per_photon / electron_per_dn
            gain = dn_per_photon
            read_noise = 1.15
        elif 'lasco' in filename:
            electron_per_photon = 1
            electron_per_dn = 15
            dn_per_photon = electron_per_photon / electron_per_dn
            gain = dn_per_photon
            read_noise = 0.3
        else:
            dn_per_photon = 1
        image /= dn_per_photon
    else:
        image = cv2.imread(filename)
        image = np.float32(image.sum(axis=2))
        image = np.flipud(image)
        header = None

    return image, header, gain, read_noise


class Image:
    def __init__(self, filename, roi=None):
        self.filename = filename
        self._data = None
        self._noise = None
        self._header = None
        if roi is not None:
            self.roi = slice(roi[1], roi[3]), slice(roi[0], roi[2])
        else:
            self.roi = None
        self.reference = None

    def __array__(self):
        return self.data

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
            with fits.open(self.filename) as hdul:
                self._header = hdul[-1].header
        return self._header

    @header.setter
    def header(self, header):
        self._header = header

    @property
    def noise(self):
        if self._noise is None:
            image = np.copy(self.data)
            image[image < 0] = 0
            self._noise = np.sqrt(self.header['GAIN'] * image + self.header['RDNOISE'] ** 2)
            self._noise[self._noise <= 0] = 1
        return self._noise

    def read(self):
        filename = self.filename
        image, header, gain, read_noise = read(filename)
        header['GAIN'] = gain
        header['RDNOISE'] = read_noise
        if self._data is None:
            if self.roi is None:
                self.roi = slice(0, image.shape[1]), slice(0, image.shape[0])
        self._data = image
        self.header = header
        self.crop()

    def crop(self):
        self.header['CRPIX1'] -= self.roi[1].start
        self.header['CRPIX2'] -= self.roi[0].start
        self.data = self.data[self.roi]

    def register(self, x0=0, y0=0, order=3, opencv=False):

        if self.reference:
            x0, y0 = self.reference.header["CRVAL1"] / self.reference.header["CDELT1"],\
                     self.reference.header["CRVAL2"] / self.reference.header["CDELT2"]
        x, y = self.header["CRVAL1"] / self.header["CDELT1"], self.header["CRVAL2"] / self.header["CDELT2"]
        euclidian = EuclidianTransform(x - x0, y - y0, 0, 1, direction='inverse')
        rectifier = Rectifier(euclidian)

        self._data = rectifier(self.data, self.data.shape,
                               (0, self.data.shape[1] - 1), (0, self.data.shape[0] - 1), order=order, opencv=opencv)

    def geometric_rectification(self, target=None, north_up=True, center=True, order=2, opencv=True):

        if target is None:
            if center:
                target = (0, 0)
            else:
                target = self.header['CRVAL1'], self.header['CRVAL2']

        alpha = -(np.radians((self.header['CRVAL1'] - target[0])/3600))
        beta = np.radians((self.header['CRVAL2'] - target[1])/3600)

        gamma = -np.radians(self.header['CROTA']) if north_up else 0

        # x, y in the plane of the sky, z towards the observer
        # yaw = around y, pitch = around x, roll = around z
        Ry = rotationmatrix(alpha, 1)
        Rx = rotationmatrix(beta, 2)
        Rz = rotationmatrix(gamma, 0)

        # K & K^{-1}
        Km1 = np.array([[np.radians(self.header['CDELT1']/3600), 0, -(self.header['CRPIX1']-1)*np.radians(self.header['CDELT1']/3600)],
                        [0, np.radians(self.header['CDELT2']/3600), -(self.header['CRPIX2']-1)*np.radians(self.header['CDELT2']/3600)],
                        [0, 0, 1]])
        K = np.array([[1/np.radians(self.header['CDELT1']/3600), 0, self.header['CRPIX1']-1],
                      [0, 1/np.radians(self.header['CDELT2']/3600), self.header['CRPIX2']-1],
                      [0, 0, 1]])
        KH = K @ Rz @ Rx @ Ry @ Km1
        KH /= KH[2, 2]

        transform = HomographicTransform(KH, dtype=np.float32)

        rectifier = Rectifier(transform)

        xfov = 0, self.data.shape[1]-1
        yfov = 0, self.data.shape[0]-1

        self.data = rectifier(self.data, self.data.shape, xfov, yfov, order=order, opencv=opencv)

        if center:
            self.header['CRVAL1'] = 0
            self.header['CRVAL2'] = 0
        if north_up:
            self.header['PC1_1'] = 1
            self.header['PC1_2'] = 0
            self.header['PC2_1'] = 0
            self.header['PC2_2'] = 1
            self.header['CROTA'] = 0


class Cube:
    def __init__(self, filenames, roi=None):
        self.filenames = filenames
        self._data = None
        self._noise = None
        self._header = None
        if roi is not None:
            self.roi = slice(roi[1], roi[3]), slice(roi[0], roi[2])
        else:
            self.roi = None
        self.reference = None

    def __array__(self):
        return self.data

    @property
    def data(self):
        if self._data is None:
            self.read()
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def noise(self):
        pass

    def read(self):
        filenames = [self.filenames] if type(self.filenames) is str else self.filenames
        for i, filename in enumerate(filenames):
            image, header, gain, read_noise = read(filename)
            header['GAIN'] = gain
            header['RDNOISE'] = read_noise
            if self._data is None:
                if self.roi is None:
                    self.roi = slice(0, image.shape[1]), slice(0, image.shape[0])
                shape = (len(filenames), self.roi[0].stop - self.roi[0].start, self.roi[1].stop - self.roi[1].start)
                self._data = np.empty_like(image, shape=shape)
                self.header = header
            self._data[i] = image
            self.header.append(header)
        if len(filenames) == 1:
            self._data = self._data.squeeze()
            # self.header = self.header[0]
        self.crop()

    def crop(self):
        pass

    def register(self, x0=0, y0=0, order=3, opencv=False):
        pass

    def geometric_rectification(self, north_up=True, center=True, delta_alpha=0, delta_beta=0, order=2, opencv=True):
        pass
