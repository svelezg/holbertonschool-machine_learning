#!/usr/bin/env python3
""" contains the HP function"""
import numpy as np


def HP(Di, beta):
    """
    calculates the Shannon entropy and P affinities relative to a data point:
    :param Di: numpy.ndarray of shape (n - 1,)
        containing the pariwise distances between a data point
        and all other points except itself
        n is the number of data points
    :param beta: beta value for the Gaussian distribution
    :return: (Hi, Pi)
        Hi: the Shannon entropy of the points
        Pi: a numpy.ndarray of shape (n - 1,)
            containing the P affinities of the points
    """
    num = np.exp(-Di * beta)
    den = np.sum(num)

    Pi = num / den

    Hi = - np.sum(Pi * np.log2(Pi))

    return Hi, Pi
