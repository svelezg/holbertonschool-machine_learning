#!/usr/bin/env python3
""" contains the P_affinities function"""
import numpy as np


def Q_affinities(Y):
    """
    calculates the Q affinities
    :param Y: numpy.ndarray of shape (n, ndim)
        containing the low dimensional transformation of X
        n is the number of points
        ndim is the new dimensional representation of X
    :return: Q, num
        Q is a numpy.ndarray of shape (n, n) containing the Q affinities
        num is a numpy.ndarray of shape (n, n)
            containing the numerator of the Q affinities
    """
    n, ndims = Y.shape

    # (a - b)**2 = a^2 - 2ab + b^2 expansion
    a2 = np.sum(Y ** 2, axis=1)
    b2 = np.sum(Y ** 2, axis=1)[:, np.newaxis]
    ab = np.matmul(Y, Y.T)
    D = a2 - 2 * ab + b2

    # Student t-distribution with one degree of freedom
    # (which is the same as a Cauchy distribution)
    num = (1 + D) ** (-1)

    np.fill_diagonal(num, 0.)

    den = np.sum(num)

    Q = num / den

    return Q, num
