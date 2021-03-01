import lmfit

from ..diffusion_eqs import sphere_constant as eqs


def conc_constant_global_wrapper(params, data, t, x):
    data_theo = eqs.conc_constant_global(
        t, x,
        params["D_over_a_sq"], params["t0"],
    )
    return (data[..., 0]-data_theo) / data[..., 1]


def conc_constant_global(data, t, x, start_t0):
    params = lmfit.Parameters()
    params.add("D_over_a_sq", value=0.01, min=0, vary=True)
    params.add("t0", value=start_t0, vary=True)

    mini = lmfit.Minimizer(
        conc_constant_global_wrapper, params,
        fcn_args=(data, t, x),
    )

    result = mini.minimize()
    ci_out = lmfit.conf_interval(mini, result)
    return result, ci_out


def conc_growing_global_wrapper(params, data, t, x):
    params = params.valuesdict()
    data_theo = eqs.conc_growing_global(
        t, x,
        params["D_over_a_sq"], params["beta"], params["t0"],
    )
    return (data[..., 0]-data_theo) / data[..., 1]


def conc_growing_global(data, t, x, start_t0, beta=None):
    params = lmfit.Parameters()
    params.add("D_over_a_sq", value=0.01, min=0, vary=True)
    if beta is None:
        params.add("beta", value=0.1, min=0, vary=True)
        params.add("t0", value=start_t0, vary=True)
    else:
        params.add("beta", value=beta, vary=False)
        params.add("t0", value=start_t0, vary=False)

    mini = lmfit.Minimizer(
        conc_growing_global_wrapper, params,
        fcn_args=(data, t, x),
    )

    result = mini.minimize()
    ci_out = lmfit.conf_interval(mini, result)
    return result, ci_out
