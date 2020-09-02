#!/usr/bin/env python3
"""contains the GRUCell class"""

import numpy as np


def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """softmax function"""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class GRUCell:
    """
    class GRUCell that represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        constructor
        :param i: dimensionality of the data
        :param h: dimensionality of the hidden state
        :param o: dimensionality of the outputs
        """
        # Weights of the cell
        # vertical stacking for Wz, Wr and Wh
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        # Biases of the cell
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        :param h_prev: numpy.ndarray of shape (m, h)
            containing the previous hidden state
        :param x_t: numpy.ndarray of shape (m, i)
            that contains the data input for the cell
            m is the batch size for the data
        :return: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        # horizontal stacking of previous hidden state and input
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # 1. Update gate (z)
        z_t = sigmoid(np.matmul(h_x, self.Wz) + self.bz)

        # 2. Reset gate (r)
        r_t = sigmoid(np.matmul(h_x, self.Wr) + self.br)

        # 3. Current memory content
        h_x = np.concatenate((r_t * h_prev, x_t), axis=1)

        # input for tanh activation
        Z_next = np.matmul(h_x, self.Wh) + self.bh

        # tanh activation outputting hidden activated state
        h_prime = np.tanh(Z_next)

        # 4. Final memory at current time step
        h_next = (1 - z_t) * h_prev + z_t * h_prime

        # input for softmax activation
        Z_y = np.matmul(h_next, self.Wy) + self.by

        # softmax activation outputting GRUCell final output
        y = softmax(Z_y)

        return h_next, y
