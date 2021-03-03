"""
These functions implement some of the diffusion equations derived in J. Crank,
The Mathematics of Diffusion, 1975, Chapter 6 - Diffusion in a sphere.
"""
import numpy as np


def summand(t, n, D_over_a_sq, r_over_a):
    mt, mn = np.meshgrid(t, n)
    return np.sin(
        mn * np.pi * r_over_a
    ) * np.exp(
        -mn**2 * np.pi**2 * D_over_a_sq * mt
    ) / mn


def summand_0(t, n, D_over_a_sq):
    mt, mn = np.meshgrid(t, n)
    return np.exp(-mn**2 * np.pi**2 * D_over_a_sq * mt)


def conc_constant_global(
    t, r_over_as,
    D_over_a_sq, t0,
    n_max=201, min_t=1e-6
):
    result = np.empty((len(t), len(r_over_as)))
    for i, r_over_a in enumerate(r_over_as):
        result[:, i] = conc_constant(
            t, D_over_a_sq, r_over_a, t0,
            n_max=n_max, min_t=min_t,
        )
    return result


def conc_growing_global(
    t, r_over_as,
    D_over_a_sq, beta, t0,
    n_max=201, min_t=1e-6
):
    result = np.empty((len(t), len(r_over_as)))
    for i, r_over_a in enumerate(r_over_as):
        result[:, i] = conc_growing(
            t, D_over_a_sq, r_over_a, beta, t0,
            n_max=n_max, min_t=min_t,
        )
    return result


def conc_constant(t, D_over_a_sq, r_over_a, t0, n_max=201, min_t=1e-6):
    """
    Equation 6.18 with a time offset t0.
    """
    t_tmp = np.where(t-t0 > min_t, t-t0, min_t)

    if n_max % 2 == 0:
        odd_start = 1
        even_start = 0
    else:
        odd_start = 0
        even_start = 1
    n_max += 1
    n_odd = np.arange(1, n_max, 2)
    n_even = np.arange(2, n_max, 2)

    sums = np.empty((n_max-1, len(t_tmp)))

    if r_over_a > 0:
        sums[odd_start::2, :] = -summand(
            t_tmp, n_odd, D_over_a_sq, r_over_a
        )[::-1, :]
        sums[even_start::2, :] = summand(
            t_tmp, n_even, D_over_a_sq, r_over_a
        )[::-1, :]

        sum_ = sums.cumsum(axis=0)[-1, :]
        sum_ = 1 + 2/r_over_a/np.pi*sum_
    else:
        sums[odd_start::2, :] = -summand_0(t_tmp, n_odd, D_over_a_sq)[::-1, :]
        sums[even_start::2, :] = summand_0(t_tmp, n_even, D_over_a_sq)[::-1, :]

        sum_ = sums.cumsum(axis=0)[-1, :]
        sum_ = 1 + 2*sum_
    return np.where(t-t0 > min_t, sum_, 0)


def total_constant(t, D_over_a_sq, t0, n_max=201, min_t=1e-6):
    """
    Equation 6.20
    """
    t_tmp = np.where(t-t0 > min_t, t-t0, min_t)

    ns = np.arange(1, n_max)
    mt, mn = np.meshgrid(t_tmp, ns)

    sums = np.exp(-mn**2 * np.pi**2 * D_over_a_sq * mt) / mn**2

    return 1 - 6/np.pi**2 * sums[::-1].cumsum(axis=0)[-1, :]


def summand_g(t, n, D_over_a_sq, r_over_a, beta):
    mt, mn = np.meshgrid(t, n)
    return np.sin(
        mn * np.pi * r_over_a
    ) * np.exp(
        -mn**2 * np.pi**2 * D_over_a_sq * mt
    ) / mn / (mn**2 * np.pi**2 - beta / D_over_a_sq)


def summand_g0(t, n, D_over_a_sq, beta):
    mt, mn = np.meshgrid(t, n)
    return np.exp(
        -mn**2 * np.pi**2 * D_over_a_sq * mt
    ) / (mn**2 * np.pi**2 - beta / D_over_a_sq)


def conc_growing(t, D_over_a_sq, r_over_a, beta, t0, n_max=201, min_t=1e-6):
    """
    Equation 6.25
    """
    assert r_over_a <= 1., f"r_over_a value {r_over_a} > 1"
    t_tmp = np.where(t-t0 > min_t, t-t0, min_t)

    if n_max % 2 == 0:
        odd_start = 1
        even_start = 0
    else:
        odd_start = 0
        even_start = 1
    n_max += 1
    n_odd = np.arange(1, n_max, 2)
    n_even = np.arange(2, n_max, 2)

    sums = np.empty((n_max-1, len(t_tmp)))

    if r_over_a > 0:
        first_p = np.exp(-beta * t_tmp) * np.sin(
            np.sqrt(beta / D_over_a_sq) * r_over_a
        ) / np.sin(np.sqrt(beta / D_over_a_sq)) / r_over_a

        sums[odd_start::2, :] = -summand_g(
            t_tmp, n_odd, D_over_a_sq, r_over_a, beta
        )[::-1, :]
        sums[even_start::2, :] = summand_g(
            t_tmp, n_even, D_over_a_sq, r_over_a, beta
        )[::-1, :]

        sum_ = sums.cumsum(axis=0)[-1, :]
        sum_ = 1 - first_p - 2*beta/D_over_a_sq/r_over_a/np.pi*sum_
    else:
        raise NotImplemented("r_over_a == 0 is buggy")
        sums[odd_start::2, :] = -summand_g0(
            t_tmp, n_odd, D_over_a_sq, beta
        )[::-1, :]
        sums[even_start::2, :] = summand_g0(
            t_tmp, n_even, D_over_a_sq, beta
        )[::-1, :]

        sum_ = sums.cumsum(axis=0)[-1, :]
        sum_ = 1 - np.exp(-beta * t_tmp) - 2*beta/D_over_a_sq*sum_
    return np.where(t-t0 > min_t, sum_, 0)
