#!/usr/bin/env python3
"""contains the mean_cov function"""

import numpy as np


def mean_cov(X):
    """
    calculates the mean and covariance of a data set
    :param X: numpy.ndarray of shape (n, d) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
    :return: mean, cov:
        mean is a numpy.ndarray of shape (1, d)
            containing the mean of the data set
        cov is a numpy.ndarray of shape (d, d)
            containing the covariance matrix of the data set
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        err = 'X must be a 2D numpy.ndarray'
        raise TypeError(err)
    n, d = X.shape

    if n < 2:
        err = 'X must contain multiple data points'
        raise ValueError(err)

    mean = np.sum(X, axis=0) / n

    cov = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            cov[i, j] = np.matmul(X[:, i], X[:, j]) / (n - 1) - mean[i] * mean[j]

    return mean, cov
