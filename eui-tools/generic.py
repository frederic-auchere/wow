import os
import cv2
import numpy as np
from rectify.rectify import EuclidianTransform, Rectifier
from matplotlib import patches
from astropy.io import fits


def make_subplot(image, ax, norm,
                 title=None, x_lims=None, y_lims=None, interpolation='nearest', cmap='gray', inset=None):
    ax.imshow(image, origin='lower', norm=norm, cmap=cmap, interpolation=interpolation)
    if x_lims is None:
        x_lims = 0, image.shape[1]
    if y_lims is None:
        y_lims = 0, image.shape[0]
    if title:
        ax.text(x_lims[0]+5, y_lims[0]+5,
                title,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.5, 'pad': 1},
                ha='left', va='bottom')
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    ax.axis(False)
    if inset:
        axins = ax.inset_axes([0, 0.5, 0.5, 0.5])
        axins.imshow(image, norm=norm, origin='lower', cmap=cmap, interpolation=interpolation)
        inset_x_lims, inset_y_lims = (inset[0], inset[1]), (inset[2], inset[3])
        axins.set_xlim(*inset_x_lims)
        axins.set_ylim(*inset_y_lims)
        axins.axis('off')
        x_mid = (x_lims[1] - x_lims[0])/2
        y_mid = (y_lims[1] - y_lims[0])/2
        ax.plot([0, x_mid], [y_mid, y_mid], '-', color='white')
        ax.plot([x_mid, x_mid], [y_mid, y_lims[1]], '-', color='white')
        rect = patches.Rectangle((inset[0], inset[2]), inset[1] - inset[0], inset[3] - inset[2],
                                 linewidth=1, edgecolor='white', facecolor='none', alpha=0.5)
        ax.add_patch(rect)


def read_data(data):
    filename = str(data['file'])
    ext = os.path.splitext(filename)[1]
    if ext == '.fit' or ext == '.fits' or ext == '.fts':
        with fits.open(filename) as hdul:
            image = np.float32(hdul[-1].data)
            header = hdul[-1].header
        if 'eui-fsi' in filename:
            try:
                flat = 1 + (1.5*(np.flipud(fits.getdata(r'C:\Users\fauchere\Documents\02-Programmes\Python\local_packages\eui\data\fsi\flats\flat_fsi_HG_20170202T000000000.fts') - 1)))
                image /= flat[0:image.shape[0], 0:image.shape[1]]
            except FileNotFoundError:
                pass
            if '_L2_' in filename:
                image *= header['XPOSURE']
            if '304' in filename:
                dn_per_photon = 3.8803
            else:
                dn_per_photon = 7.3776
            data['gain'] = dn_per_photon
            data['read_noise'] = 1.5
        elif 'eui-hrieuv' in filename:
            if '_L2_' in filename:
                image *= header['XPOSURE']
            dn_per_photon = 5.2764
            data['gain'] = dn_per_photon
            data['read_noise'] = 1.5
        elif 'aia' in filename:
            electron_per_dn = 17.7
            electron_per_photon = 13.6 * 911.0 / (3.65 * 171)
            dn_per_photon = electron_per_photon/electron_per_dn
            data['gain'] = dn_per_photon
            data['read_noise'] = 1.15
        elif 'lasco' in filename:
            electron_per_photon = 1
            electron_per_dn = 15
            dn_per_photon = electron_per_photon/electron_per_dn
            data['gain'] = dn_per_photon
            data['read_noise'] = 0.3
        else:
            dn_per_photon = 1
        image /= dn_per_photon
    else:
        image = cv2.imread(filename)
        image = np.float32(image.sum(axis=2))
        image = np.flipud(image)
        header = None
    if data['roi'] is None:
        roi = [0, 0, image.shape[1], image.shape[0]]
    else:
        roi = data['roi']
    return image[roi[1]:roi[3], roi[0]:roi[2]], header


def register(img, header, reference_header, order=3, opencv=False):

    x0, y0 = reference_header["CRVAL1"] / reference_header["CDELT1"],\
             reference_header["CRVAL2"] / reference_header["CDELT2"]
    x, y = header["CRVAL1"] / header["CDELT1"], header["CRVAL2"] / header["CDELT2"]
    euclidian = EuclidianTransform(x - x0, y - y0, 0, 1, direction='inverse')
    rectifier = Rectifier(euclidian)

    return rectifier(img, img.shape, (0, img.shape[1] - 1), (0, img.shape[0] - 1), order=order, opencv=opencv)


def data_noise(image, data):
    image = np.copy(image)
    image[image < 0] = 0
    noise = np.sqrt(data['gain']*image + data['read_noise']**2)
    noise[noise <= 0] = 1
    return noise
