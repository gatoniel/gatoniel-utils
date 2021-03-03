import numpy as np

from scipy.ndimage import center_of_mass
from scipy.ndimage.morphology import distance_transform_edt


def create_position_maps(label, pixel_size=1):
    """
    Creates a flattened array for different position measures corresponding
    to the pixels belonging to an object in an image.
    Arguments:
        - label: binary image describing the object
        - pixel_size: pixel_size of image
    Returns:
        - pos_measures_flat: flattened array of shape (num_object_pixels,
        len(info)). pos_measures[i, j] is the j-th position measure for the
        i-th pixel of the object.
        - pos_measures_img: img of shape of label with len(info) channels
        in the last axis. Non-object pixels have pos measure 0.
        - info: list of tuples with info about the name of the different
        position measures that are saved in the last axis of pos_measures.
        info[i][0]: abbreviation for position measure i
        info[i][1]: long info for position measure i
    """
    assert label.dtype == bool
    edt = distance_transform_edt(label)
    r_mass = distance_transform_radial(label)
    # transform edt to make it more comparable to radial distances
    edt_inv = edt.max() - edt

    # finally, we will only use the pixels that belong to label
    pos_measures_flat = np.stack([
        edt[label],
        r_mass[label],
        edt_inv[label],
    ], axis=-1)
    pos_measures_flat *= pixel_size

    pos_measures_img = np.zeros((
        label.shape[0], label.shape[1], pos_measures_flat.shape[1]
    ))
    pos_measures_img[label, :] = pos_measures_flat

    return pos_measures_flat, pos_measures_img


def create_position_maps_info():
    return [
        ("edt", "euclidean distance transform"),
        ("r_mass", "radial distance from center of mass"),
        ("edt_inv", "euclidean distance transform inverted"),
    ]


def distance_transform_radial(label):
    x = np.arange(label.shape[1])
    y = np.arange(label.shape[0])
    my, mx = np.meshgrid(y, x, indexing="ij")
    mass_center = center_of_mass(label)
    r_mass = np.sqrt((my - mass_center[0])**2 + (mx - mass_center[1])**2)
    r_mass[np.logical_not(label)] = 0
    return r_mass
