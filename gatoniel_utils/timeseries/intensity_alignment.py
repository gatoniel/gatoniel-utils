import numpy as np


def align_intensity_jumps(
    means, intensity_jump, detection_value, num_smooth=5
):
    adjusted = np.copy(means)
    diff = np.diff(adjusted)
    align_mask = np.zeros(means.shape[0], dtype=np.bool)

    for i in range(num_smooth):
        jump_inds = diff < -detection_value

        align_mask[1:][jump_inds] = True

        adjusted[1:][jump_inds] += intensity_jump
        diff = np.diff(adjusted)
    return adjusted, align_mask
