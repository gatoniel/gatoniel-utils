import numpy as np


def bin_x_from_y(x, y, num_bins=100):
    _, bins = np.histogram(y, num_bins)

    y_ = (bins[:-1] + bins[1:]) / 2

    x_ = x_from_bins(x, y, bins)
    return x_, y_


def x_from_bins(x, y, bins):
    assert x.ndim <= 3
    x = np.at least_3d(x)

    bins = np.copy(bins)
    bins[0] -= 1
    bins[-1] += 1
    x_ = np.empty((x.shape[0], len(bins) - 1, x.shape[2], 2))

    indices = np.digitize(y, bins)
    assert indices.max() == len(bins) - 1
    for i in range(len(bins) - 1):
        tmp_inds = indices == i + 1
        x_[:, i, :, 0] = x[:, tmp_inds, :].mean(axis=1)
        x_[:, i, :, 1] = x[:, tmp_inds, :].std(axis=1)
    return np.squeeze(x_)
