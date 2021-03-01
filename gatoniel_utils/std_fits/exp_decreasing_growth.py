import numpy as np
import lmfit


def func(t, beta, t0, a, b):
    y = a * (1 - np.exp(-beta * (t - t0))) + b
    y[t < t0] = b
    return y


def func_wrapper(params, data, t):
    params = params.valuesdict()
    data_theo = func(
        t, params["beta"], params["t0"], params["a"], params["b"]
    )
    return (data[..., 0] - data_theo) / data[..., 1]


def fit(data, t, beta=1, t0=0, a=None, b=None):
    params = lmfit.Parameters()
    params.add("beta", value=beta, min=0, vary=True)
    params.add("t0", value=t0, vary=True)

    if a is None:
        params.add("a", value=1, vary=False)
    else:
        params.add("a", value=a, min=0, vary=True)
    if b is None:
        params.add("b", value=0, vary=False)
    else:
        params.add("b", value=b, vary=True)

    mini = lmfit.Minimizer(
        func_wrapper, params,
        fcn_args=(data, t),
    )

    result = mini.minimize()
    return result
