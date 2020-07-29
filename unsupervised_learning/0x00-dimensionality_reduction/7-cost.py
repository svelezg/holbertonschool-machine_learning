#!/usr/bin/env python3
""" contains the cost function"""
import numpy as np


def cost(P, Q):
    """
    calculates the cost of the t-SNE transformation
    :param P: numpy.ndarray of shape (n, n) containing the P affinities
    :param Q: numpy.ndarray of shape (n, n) containing the Q affinities
    :return: C, the cost of the transformation
    """
    almost = np.array([[1e-12]])

    argument = P / np.maximum(Q, almost)
    log_ar = np.log(np.maximum(argument, almost))
    product = (P * log_ar)

    C = np.sum(product)

    return C
