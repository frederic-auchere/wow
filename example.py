from wow import wow
from wow.plotting import make_subplot
from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, PowerStretch, PercentileInterval
import matplotlib.pyplot as plt

# Loads a sample Solar Orbiter/EUI/HRI_EUV image
sample_file = r'sample_data\solo_L2_eui-hrieuv174-image_20220317T032000234_V01.fits'
source_image = fits.getdata(sample_file)

wow_image = wow(source_image)  # Basic WOW, non-bilateral, non de-noised. Fast.

# Denoise in the first two scales at 5 and 1 sigma above the local noise level
denoise_coefficients = [5, 1]
denoised_wow = wow(source_image, denoise_coefficients=denoise_coefficients)
# No de-noising, but use the edge-aware transform (slower)
bilateral_wow = wow(source_image, bilateral=1)
# Same with de-noising
denoised_bilateral_wow = wow(source_image, denoise_coefficients=denoise_coefficients, bilateral=1)
# Same merged with a gamma-stretched image
gamma_denoised_bilateral_wow = wow(source_image, denoise_coefficients=denoise_coefficients, bilateral=1, h=0.995)

power_stretch = PowerStretch(1/2.5)  # Used for the original image
linear_stretch = LinearStretch()     # Used for all the other images
interval = PercentileInterval(99.9)  # Used for all images

images = (source_image, wow_image, denoised_wow, bilateral_wow, denoised_bilateral_wow, gamma_denoised_bilateral_wow)
stretches = (power_stretch, linear_stretch, linear_stretch, linear_stretch, linear_stretch, linear_stretch)
titles = ('original',
          'WOW',
          'de-noised WOW',
          'bilateral WOW',
          'de-noised bilateral WOW',
          r'$\gamma$-scaled + de-noised bilateral WOW')

roi = [1300, 1650, 150, 500]  # coordinates of the inset zoom

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, image, stretch, title in zip(axes.flatten(), images, stretches, titles):
    norm = ImageNormalize(image, stretch=stretch, interval=interval)
    make_subplot(image, ax, norm, inset=roi)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
