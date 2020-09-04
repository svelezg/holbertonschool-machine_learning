#!/usr/bin/env python3
"""contains the rnn function"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    performs forward propagation for a simple RNN
    :param rnn_cells: ist of RNNCell instances of length l
        that will be used for the forward propagation
        l is the number of layers
    :param X: data to be used, given as a numpy.ndarray
        of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    :param h_0: initial hidden state, given as a numpy.ndarray
        of shape (l, m, h)
        h is the dimensionality of the hidden state
    :return: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    n_layers = len(rnn_cells)
    t, m, i = X.shape
    _, _, h_ = h_0.shape

    H = np.zeros((t + 1, n_layers, m, h_))
    H[0] = h_0

    for step in range(t):
        for layer in range(n_layers):
            if layer == 0:
                h, y = rnn_cells[layer].forward(H[step, layer], X[step])
            else:
                h, y = rnn_cells[layer].forward(H[step, layer], h)
            H[step+1, layer, ...] = h

            # final layer
            if layer == n_layers - 1:
                if step == 0:
                    Y = y
                else:
                    Y = np.concatenate((Y, y))

    Y = Y.reshape(t, m, Y.shape[-1])

    return H, Y
