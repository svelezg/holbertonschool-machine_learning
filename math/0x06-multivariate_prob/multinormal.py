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

    def pdf(self, x):
        """
        calculates the PDF at a data point
        :param x: numpy.ndarray of shape (d, 1)
            containing the data point whose PDF should be calculated
        :return:  value of the PDF
        """
        if not isinstance(x, np.ndarray):
            err = 'x must by a numpy.ndarray'
            raise TypeError(err)
        d, n = x.shape

        if n != 1:
            err = 'x must have the shape ({d}, 1)'.format(d)
            raise ValueError(err)

        det = np.linalg.det(self.cov)
        den = np.sqrt(((2 * np.pi) ** d) * det)

        dev = (x - self.mean).T
        inv = np.linalg.inv(self.cov)

        inner = np.matmul(dev, inv)
        outer = np.matmul(inner, dev.T)

        pdf = np.exp((-1/2) * outer) / den

        return pdf[0, 0]
