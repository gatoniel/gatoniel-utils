def normalize_start_end(x, inds=20):
    mi = x[:inds, :].mean(axis=0)
    ma = x[-inds:, :].mean(axis=0)
    return (x - mi) / (ma - mi)
