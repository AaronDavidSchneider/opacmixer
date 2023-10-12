"""Module housing several different input and output scaling functions.
 The defaults are at the end of the file with the prefix default_"""
import numpy as np


def diff(vals, do_log=False):
    """
    Calculating the difference along $g$ values

    Parameters
    ----------
    vals: array like
        the values to be transformed
    do_log: bool
        transform to log space

    Returns
    -------
    trans: array like
        The transformed values
    """
    diffvals = np.diff(vals, axis=1)
    zero_val = vals[:, 0]
    if do_log:
        diffvals = np.log(diffvals + 1e-45)
        zero_val = np.log(vals[:, 0] + 1e-45)

    if len(vals.shape) == 3:
        return np.concatenate((zero_val[:, None, :], diffvals), axis=1)

    return np.concatenate((zero_val[:, None], diffvals), axis=-1)


def integrate_diff(diff_vals, do_log=False):
    """
    Reverting diff

    Parameters
    ----------
    diff_vals: array like
        the values to be transformed
    do_log: bool
        transform to log space

    Returns
    -------
    int_res: array like
        The transformed values
    """

    rev_diff = diff_vals
    if do_log:
        rev_diff = np.exp(diff_vals) - 1e-45

    intdiff = np.cumsum(rev_diff[:, 1:], axis=-1) + rev_diff[:, 0][:, None]
    int_res = np.concatenate((rev_diff[:, 0][:, None], intdiff), axis=-1)
    return int_res


def transform_x_scaled(X):
    """
    Scale by sum along $g$

    Parameters
    ----------
    X: array like
        input values, needs to have $g$ at last axis

    Returns
    -------
    X_scaled: array like
        the scaled X values
    """
    xlargest = X.sum(axis=-1)[:, -1]
    return X / xlargest[:, None, None]


def transform_y_scaled(X, y):
    """
    Scale y by sum of X along $g$

    Parameters
    ----------
    X: array like
        input values, needs to have $g$ at last axis
    y: array like
        the targets which are to be scaled

    Returns
    -------
    y_scaled: array like
        the scaled y values
    """
    xlargest = X.sum(axis=-1)[:, -1]
    return y / xlargest[:, None]


def inverse_transform_y_scaled(X, y):
    """
    revert transform_y_scaled

    Parameters
    ----------
    X: array like
        input values, needs to have $g$ at last axis
    y: array like
        the scaled targets which are to be reverted

    Returns
    -------
    y_scaled: array like
        the transformed y values
    """
    xlargest = X.sum(axis=-1)[:, -1]
    return y * xlargest[:, None]


def transform_x_sum(X, do_log=True):
    """
    X-Xsum, where X sum is the sum along $g$ values

    Parameters
    ----------
    X: array like
        the input data. Last axis needs to be $g$
    do_log: bool
        take log of the scaled input data

    Returns
    -------
    transformed: array like
        the transformed input
    """
    xsum = X.sum(axis=-1)
    if do_log:
        transformed = np.log(X / xsum[:, :, None])
    else:
        transformed = X - xsum[:, :, None]

    return transformed


def transform_y_sum(X, y, do_log=True):
    """
    y-Xsum, where X sum is the sum along $g$ values

    Parameters
    ----------
    X: array like
        the input data. Last axis needs to be $g$
    y: array like
        the targets to be scaled
    do_log: bool
        take log of the scaled input data

    Returns
    -------
    transformed: array like
        the scaled targets
    """
    xsum = X.sum(axis=-1)
    if do_log:
        transformed = np.log(y / xsum)
    else:
        transformed = y - xsum
    return transformed


def inverse_transform_y_sum(X, y, do_log=True):
    """
    Revert transform_y_sum

    Parameters
    ----------
    X: array like
        the input data. Last axis needs to be $g$
    y: array like
        the scaled targets to be transformed
    do_log: bool
        take log of the scaled input data

    Returns
    -------
    trans: array like
        transformed targets
    """
    xsum = X.sum(axis=-1)
    if do_log:
        return np.exp(y) * xsum

    return y + xsum


def transform_x_diff(X, do_log=True):
    """
    Scales input using the diff

    Parameters
    ----------
    X: array like
        The input data
    do_log: bool
        go to logspace

    Returns
    -------
    trans: array like
        The scaled input data
    """
    return diff(X, do_log=do_log)


def transform_y_diff_sum(X, y, do_log=True):
    """
    Scales targets using the diff

    Parameters
    ----------
    X: array like
        The input data
    y: array like
        The targets to be scaled
    do_log: bool
        go to logspace

    Returns
    -------
    trans: array like
        The scaled targets
    """
    xsum = X.sum(axis=-1)
    if do_log:
        return diff(y, do_log=True) - diff(xsum, do_log=True)

    return diff((y - xsum) / (xsum[:, -1][:, None] + 1), do_log=False)


def inverse_transform_y_diff_sum(X, y, do_log=True):
    """
    Revert transform_y_diff_sum

    Parameters
    ----------
    X: array like
        The input data
    y: array like
        The scaled targets to transform
    do_log: bool
        Go to log space

    Returns
    -------
    trans: array like
        the recovered targets
    """
    xsum = X.sum(axis=-1)
    if do_log:
        return integrate_diff(y + diff(xsum, do_log=True), do_log=True)

    return (xsum[:, -1][:, None] + 1) * integrate_diff(y, do_log=False) + xsum


def default_input_scaling(X):
    """
    Default function used for input scaling

    Parameters
    ----------
    X: array like
        The input data to be scaled

    Returns
    -------
    trans: array like
        The transformed input data
    """
    return transform_x_sum(X, do_log=True)


def default_output_scaling(X, y):
    """
    Default function used for output scaling

    Parameters
    ----------
    X: array like
        The input data
    y: array like
        The targets to be scaled

    Returns
    -------
    trans: array like
        The transformed targets
    """
    return transform_y_sum(X, y, do_log=True)


def default_inv_output_scaling(X, y):
    """
    Default function used to recover output scaling

    Parameters
    ----------
    X: array like
        The input data
    y: array like
        The scaled targets to be transformed

    Returns
    -------
    trans: array like
        The recovered targets
    """
    return inverse_transform_y_sum(X, y, do_log=True)
