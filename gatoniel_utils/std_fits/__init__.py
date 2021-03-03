import numpy as np
import pandas as pd


def result_to_params(res, data, res_name="fit", params_list=None):
    params_dict = {}
    if params_list is None:
        params_list = res.params.keys()
    params_np = np.empty((len(params_list), 2))

    for i, p in enumerate(params_list):
        val = res.params[p].value
        err = res.params[p].stderr

        params_np[i, 0] = val
        params_np[i, 1] = err

        params_dict[p] = val
        params_dict[p + "_std"] = err

    params_dict["rmse"] = np.sqrt(res.chisqr / np.prod(data.shape))
    params_dict["chisqr"] = res.chisqr
    params_dict["reduced chisqr"] = res.redchi

    params_pd = pd.DataFrame(params_dict, index=[res_name])
    return params_pd, params_np


def results_to_params(res_list, data, res_name_list, params_list=None):
    params_np = []
    params_pd = []
    for r_l, r_n_l in zip(res_list, res_name_list):
        pd, np = result_to_params(r_l, data, r_n_l, params_list)
        params_np.append(np)
        params_pd.append(pd)
    params_np = np.stack(params_np, axis=0)
    params_pd = pd.concat(params_pd)
    return params_pd, params_np
