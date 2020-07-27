#!/usr/bin/env python3
""" contain the multinormal class"""

import numpy as np


class MultiNormal:
    """
    Multinormal class
    """

    def __init__(self, data):
        """
        constructor
        :param data: numpy.ndarray of shape (d, n) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            err = 'data must be a 2D numpy.ndarray'
            raise TypeError(err)
        d, n = data.shape

        if n < 2:
            err = 'data must contain multiple data points'
            raise ValueError(err)

        mean = np.sum(data, axis=1) / n
        self.mean = np.expand_dims(mean, axis=1)

        deviaton = data - self.mean

        self.cov = np.matmul(deviaton, deviaton.T) / (n - 1)
