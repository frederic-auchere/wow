import os
import cv2
import numpy as np
from astropy.io import fits
from rectify.rectify import EuclidianTransform, Rectifier


class Image:
    def __init__(self, filename, roi=None):
        self.filename = filename
        self._data = None
        self._noise = None
        self.header = None
        self.roi = roi
        self.gain = None
        self.read_noise = None
        self.reference = None

    def __array__(self):
        return self.data

    @property
    def data(self):
        if self._data is None:
            self.read()
        return self._data

    @property
    def noise(self):
        if self._noise is None:
            image = np.copy(self.data)
            image[image < 0] = 0
            self._noise = np.sqrt(self.gain * image + self.read_noise ** 2)
            self._noise[self._noise <= 0] = 1
        return self._noise

    def read(self):
        filename = self.filename
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
                self.gain = dn_per_photon
                self.read_noise = 1.5
            elif 'eui-hrieuv' in filename:
                if '_L2_' in filename:
                    image *= header['XPOSURE']
                dn_per_photon = 5.2764
                self.gain = dn_per_photon
                self.read_noise = 1.5
            elif 'aia' in filename:
                electron_per_dn = 17.7
                electron_per_photon = 13.6 * 911.0 / (3.65 * 171)
                dn_per_photon = electron_per_photon / electron_per_dn
                self.gain = dn_per_photon
                self.read_noise = 1.15
            elif 'lasco' in filename:
                electron_per_photon = 1
                electron_per_dn = 15
                dn_per_photon = electron_per_photon / electron_per_dn
                self.gain = dn_per_photon
                self.read_noise = 0.3
            else:
                dn_per_photon = 1
            image /= dn_per_photon
        else:
            image = cv2.imread(filename)
            image = np.float32(image.sum(axis=2))
            image = np.flipud(image)
            header = None
        if self.roi is None:
            self.roi = [0, 0, image.shape[1], image.shape[0]]

        self._data = image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        self.header = header

    def register(self, x0=0, y0=0, order=3, opencv=False):

        if self.reference:
            x0, y0 = self.reference.header["CRVAL1"] / self.reference.header["CDELT1"],\
                     self.reference.header["CRVAL2"] / self.reference.header["CDELT2"]
        x, y = self.header["CRVAL1"] / self.header["CDELT1"], self.header["CRVAL2"] / self.header["CDELT2"]
        euclidian = EuclidianTransform(x - x0, y - y0, 0, 1, direction='inverse')
        rectifier = Rectifier(euclidian)

        self._data = rectifier(self.data, self.data.shape,
                               (0, self.data.shape[1] - 1), (0, self.data.shape[0] - 1), order=order, opencv=opencv)
