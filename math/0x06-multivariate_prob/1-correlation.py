#!/usr/bin/env python3
"""contains the correlation function"""

import numpy as np


def correlation(C):
    """
    calculates a correlation matrix
    :param C: numpy.ndarray of shape (d, d) containing a covariance matrix:
        d is the number of dimensions
    :return: numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        err = 'C must be a numpy.ndarray'
        raise TypeError(err)

    C_shape = C.shape
    if len(C_shape) != 2 or C_shape[0] != C_shape[1]:
        err = 'C must be a 2D square matrix'
        raise ValueError(err)

    diag = np.diag(C)
    diag_vect = np.expand_dims(diag, axis=0)

    std = np.sqrt(diag_vect)

    mult = np.matmul(std.T, std)

    corr = C / mult

    return corr
