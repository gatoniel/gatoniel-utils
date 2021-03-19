def normalize_start_end(x, inds=20):
    mi = x[:inds, :].mean(axis=0)
    ma = x[-inds:, :].mean(axis=0)
    return (x - mi) / (ma - mi)


def normalize_end_start(x, inds=20):
    ma = x[:inds, :].mean(axis=0)
    mi = x[-inds:, :].mean(axis=0)
    return (x - mi) / (ma - mi)
