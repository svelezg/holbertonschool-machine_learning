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

        self.mean = np.mean(data, axis=1).reshape(d, 1)

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
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # pdf formula -- multivar

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        f1 = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        f21 = -(x - self.mean).T
        f22 = np.matmul(f21, inv)
        f23 = (x - self.mean) / 2
        f24 = np.matmul(f22, f23)
        f2 = np.exp(f24)
        pdf = f1 * f2

        return pdf.reshape(-1)[0]
