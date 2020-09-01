#!/usr/bin/env python3
"""contains the rnn function"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a simple RNN
    :param bi_cell: an instance of BidirectinalCell
        that will be used for the forward propagation
    :param X: data to be used, given as a numpy.ndarray
        of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    :param h_0: initial hidden state, given as a numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state
    :param h_t: initial hidden state in the backward direction,
        given as a numpy.ndarray of shape (m, h)
    :return: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape

    H_f = []
    H_b = []
    Y = []

    # initilization
    h_f = h_0
    h_b = h_t

    H_f.append(h_0)
    H_b.append(h_t)

    # traverse inputs
    for step in range(t):
        h_f = bi_cell.forward(h_f, X[step])
        h_b = bi_cell.backward(h_b, X[t - 1 - step])

        H_f.append(h_f)
        H_b.append(h_b)

    H_f = np.array(H_f)
    H_b = [x for x in reversed(H_b)]
    H_b = np.array(H_b)
    H = np.concatenate((H_f[1:], H_b[:-1]), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
