#!/usr/bin/env python3
"""contains the RNNCell class"""

import numpy as np


class RNNCell:
    """
    class RNNCell that represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        constructor
        :param i: dimensionality of the data
        :param h: dimensionality of the hidden state
        :param o: dimensionality of the outputs
        """
        # vertical stacking of W
        self.Wh = np.random.normal(size=(i + h, h))

        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        :param h_prev: numpy.ndarray of shape (m, h)
            containing the previous hidden state
        :param x_t: numpy.ndarray of shape (m, i)
            that contains the data input for the cell
            m is the batche size for the data
        :return: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        # horizontal stacking of previous inner state and input
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # input for tanh activation
        Z_next = np.matmul(h_x, self.Wh) + self.bh

        # tanh activation outputting inner activated state
        h_next = np.tanh(Z_next)

        # input for softmax activation
        Z_y = np.matmul(h_next, self.Wy) + self.by

        # softmax activation outputting RNNCell final output
        y = np.exp(Z_y)/np.sum(np.exp(Z_y), axis=1, keepdims=True)

        return h_next, y
