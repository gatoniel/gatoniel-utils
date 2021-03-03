import numpy as np
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation


def register_timeseries(imgs, max_shift):
    registered_imgs = np.empty_like(imgs)

    registered_imgs[0, ...] = imgs[0, ...]

    for i in range(1, imgs.shape[0]):
        reference_image = registered_imgs[i - 1, ...]
        moving_image = imgs[i, ...]

        shift, error, diffphase = phase_cross_correlation(
            reference_image, moving_image, upsample_factor=100
        )
        print(shift)

        if np.all(np.abs(shift) <= max_shift):
            input_ = np.fft.fft2(moving_image)
            result = fourier_shift(input_, shift=shift)
            registered_imgs[i, ...] = np.fft.ifft2(result)
        else:
            registered_imgs[i, ...] = moving_image
    return registered_imgs
