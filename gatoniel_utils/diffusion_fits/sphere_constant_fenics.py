import lmfit
import numpy as np

from ..diffusion_eqs.sphere_constant_fenics import solve_linear_pde


def conc_growing_global_wrapper(params, data, t, x, C1=0):
    params = params.valuesdict()
    data_theo = solve_linear_pde(
        data[:, -1, 0],
        t[-1] - t[0],
        params["D_over_a_sq"],
        num_r=len(x) - 1,
        C1=C1,
    )[0]
    return (data[..., 0] - data_theo) / data[..., 1]


def conc_growing_global(data, t, x, C1=0):
    # data preprocessing:
    # c(R, t) cannot be less than zero. This might produce errors in the
    # fenics solver
    data = np.copy(data)
    data[:, -1, 0] = np.where(data[:, -1, 0] > 0, data[:, -1, 0], 0)

    params = lmfit.Parameters()
    params.add("D_over_a_sq", value=0.05, min=0, vary=True)

    mini = lmfit.Minimizer(
        conc_growing_global_wrapper,
        params,
        fcn_args=(data, t, x, C1),
    )

    result = mini.minimize()
    return result
