#!/usr/bin/env python3
"""contains the variance function"""

import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster variance for a data set
    :param X: numpy.ndarray of shape (n, d) containing the data set
    :param C: numpy.ndarray of shape (k, d)
        containing the centroid means for each cluster
    :return: var, or None on failure
    """
    k, d = C.shape

    # Eucledean norm
    # (a - b)**2 = a^2 - 2ab + b^2 expansion
    a2 = np.sum(C ** 2, axis=1)[:, np.newaxis]
    b2 = np.sum(X ** 2, axis=1)
    ab = np.matmul(C, X.T)
    D = a2 - 2 * ab + b2

    var = np.sum(np.amin(D, axis=0))

    return var
