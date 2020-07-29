#!/usr/bin/env python3
""" contains the grads function"""
import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    calculates the gradients of Y
    :param Y: numpy.ndarray of shape (n, ndim)
        containing the low dimensional transformation of X
    :param P: numpy.ndarray of shape (n, n)
        containing the P affinities of X
    :return: (dY, Q)
        dY is a numpy.ndarray of shape (n, ndim)
            containing the gradients of Y
        Q is a numpy.ndarray of shape (n, n)
            containing the Q affinities of Y
    """
    n, ndim = Y.shape

    Q, num = Q_affinities(Y)

    dY = np.zeros((n, ndim))

    f1 = P - Q
    f2 = num

    for j in range(n):
        # product of single columns out of n columns matrices
        # f1f2.shape = (n,)
        f1f2 = f1[:, j] * f2[:, j]

        # dim expansion necessary for later broadcasting
        # f1f2.shape = (n, 1)
        f1f2 = np.expand_dims(f1f2, 1)

        # difference between single point and all n points
        # Y.shape = f3.shape = (n, ndim)
        # Y[j, :].shape = (1, ndim)
        f3 = Y[j, :] - Y

        # product of f1f3 anf f3 (broadcasting)
        # product.shape = (n, ndim)
        product = f1f2 * f3

        # sum over all points
        # dY[j, :].shape = (1, ndim)
        dY[j, :] = np.sum(product, axis=0)

    return dY, Q
