from wow import wow
from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, PowerStretch, PercentileInterval
import matplotlib.pyplot as plt


sample_file = r'sample_data\solo_L2_eui-hrieuv174-image_20220317T032000234_V01.fits'
source_image = fits.getdata(sample_file)

wow_image = wow(source_image)

denoise_coefficients = [5, 1]
denoised_wow = wow(source_image, denoise_coefficients=denoise_coefficients)
bilateral_wow = wow(source_image, bilateral=1)
denoised_bilateral_wow = wow(source_image, denoise_coefficients=denoise_coefficients, bilateral=1)
gamma_denoised_bilateral_wow = wow(source_image, denoise_coefficients=denoise_coefficients, bilateral=1, h=0.995)

linear_stretch = LinearStretch()
power_stretch = PowerStretch(1/2.5)
interval = PercentileInterval(99.9)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

images = (source_image, wow_image, denoised_wow, bilateral_wow, denoised_bilateral_wow, gamma_denoised_bilateral_wow)
stretches = (power_stretch, linear_stretch, linear_stretch, linear_stretch, linear_stretch, linear_stretch)
titles = ('original',
          'WOW',
          'denoised WOW',
          'bilateral WOW',
          'denoised bilateral WOW',
          '$\gamma$-scaled + denoised bilateral WOW')
for ax, image, stretch, title in zip(axes.flatten(), images, stretches, titles):
    norm = ImageNormalize(image, stretch=stretch, interval=interval)
    ax.imshow(image, norm=norm, origin='lower', cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
