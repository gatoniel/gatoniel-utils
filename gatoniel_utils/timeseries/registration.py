import numpy as np
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation


def register_timeseries(imgs):
    registered_imgs = np.empty_like(imgs)

    registered_imgs[0, ...] = imgs[0, ...]

    for i in range(1, imgs.shape[0]):
        reference_image = registered_imgs[i - 1, ...]
        moving_image = imgs[i, ...]

        detected_shift = phase_cross_correlation(reference_image, moving_image)

        input_ = np.fft.fft2(moving_image)
        result = fourier_shift(input_, shift=detected_shift)
        reference_image[i, ...] = np.fft.ifft2(result)
    return reference_image
