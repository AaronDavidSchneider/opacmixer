import numpy as np


def diff(vals, do_log=False):
    diffvals = np.diff(vals, axis=1)
    zero_val = vals[:, 0]
    if do_log:
        diffvals = np.log(diffvals + 1e-45)
        zero_val = np.log(vals[:, 0] + 1e-45)

    if len(vals.shape) == 3:
        return np.concatenate((zero_val[:, None, :], diffvals), axis=1)
    else:
        return np.concatenate((zero_val[:, None], diffvals), axis=-1)


def integrate_diff(diff_vals, do_log=False):
    """Integrate along last axis to invert np.diff"""

    rev_diff = diff_vals
    if do_log:
        rev_diff = np.exp(diff_vals) - 1e-45

    intdiff = np.cumsum(rev_diff[:, 1:], axis=-1) + rev_diff[:, 0][:, None]
    int_res = np.concatenate((rev_diff[:, 0][:, None], intdiff), axis=-1)
    return int_res


def transform_x_scaled(X):
    xlargest = X.sum(axis=-1)[:, -1]
    return X / xlargest[:, None, None]


def transform_y_scaled(X, y):
    xlargest = X.sum(axis=-1)[:, -1]
    return y / xlargest[:, None]


def inverse_transform_y_scaled(X, y):
    xlargest = X.sum(axis=-1)[:, -1]
    return y * xlargest[:, None]


def transform_x_sum(X, do_log=True):
    xsum = X.sum(axis=-1)
    if do_log:
        transformed = np.log(X / xsum[:, :, None])
    else:
        transformed = X - xsum[:, :, None]

    return transformed


def transform_y_sum(X, y, do_log=True):
    xsum = X.sum(axis=-1)
    if do_log:
        transformed = np.log(y / xsum)
    else:
        transformed = y - xsum
    return transformed


def inverse_transform_y_sum(X, y, do_log=True):
    xsum = X.sum(axis=-1)
    if do_log:
        return np.exp(y) * xsum
    else:
        return y + xsum


def transform_x_diff(X, do_log=True):
    return diff(X, do_log=do_log)


def transform_y_diff_sum(X, y, do_log=True):
    xsum = X.sum(axis=-1)
    if do_log:
        return diff(y, do_log=True) - diff(xsum, do_log=True)
    else:
        return diff((y - xsum) / (xsum[:, -1][:, None] + 1), do_log=False)


def inverse_transform_y_diff_sum(X, y, do_log=True):
    xsum = X.sum(axis=-1)
    if do_log:
        return integrate_diff(y + diff(xsum, do_log=True), do_log=True)
    else:
        return (xsum[:, -1][:, None] + 1) * integrate_diff(y, do_log=False) + xsum


def default_input_scaling(X):
    """Default function used for input scaling"""
    return transform_x_sum(X, do_log=True)


def default_output_scaling(X, y):
    """Default function used for output scaling"""
    return transform_y_sum(X, y, do_log=True)


def default_inv_output_scaling(X, y):
    """Default function used to recover output scaling"""
    return inverse_transform_y_sum(X, y, do_log=True)
