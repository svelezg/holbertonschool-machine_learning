#!/usr/bin/env python3
"""Contains the dropout_forward_prop function"""

import numpy as np
import tensorflow as tf


def dropout_forward_prop(X, weights, L, keep_prob):
    """

    :param X: numpy.ndarray of shape (nx, m) containing
        the input data for the network
        nx is the number of input features
        m is the number of data pointsnumpy.ndarray of shape (nx, m)
        containing the input data for the network
    :param weights: dictionary of the weights and biases of the neural network
    :param L:
    :param keep_prob: probability that a node will be kept
    :return: dictionary containing the outputs of each layer and the dropout
        mask used on each layer (see example for format)
    """
    cache = {'A0': X}

    for i in range(L):
        W_key = "W{}".format(i + 1)
        b_key = "b{}".format(i + 1)
        A_key_prev = "A{}".format(i)
        A_key_forw = "A{}".format(i + 1)
        D_key = "D{}".format(i + 1)

        Z = np.matmul(weights[W_key], cache[A_key_prev]) \
            + weights[b_key]
        if i != L - 1:
            A = np.tanh(Z)
        else:
            t = np.exp(Z)
            A = (t / np.sum(t, axis=0, keepdims=True))

        if i != L-1:
            cache[D_key] = np.random.binomial(1, keep_prob, size=A.shape)
            cache[A_key_forw] = (A * cache[D_key]) / keep_prob
        else:
            cache[A_key_forw] = A

    return cache
