#!/usr/bin/env python3
"""contains the BidirectionalCell class"""

import numpy as np


def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """softmax function"""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class BidirectionalCell:
    """
    class BidirectionalCell represents a bidirectional cell of an RNN:
    """

    def __init__(self, i, h, o):
        """
        constructor
        :param i: dimensionality of the data
        :param h: dimensionality of the hidden state
        :param o: dimensionality of the outputs
        """
        # Weights of the cell
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(i + h + o, o))

        # Biases of the cell
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        :param h_prev: h_prev: numpy.ndarray of shape (m, h)
            containing the previous hidden state
        :param x_t: numpy.ndarray of shape (m, i)
            that contains the data input for the cell
        :return: h_next, c_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        # horizontal stacking of previous inner state and input
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # input for tanh activation
        Z_next = np.matmul(h_x, self.Whf) + self.bhf

        # tanh activation outputting inner activated state
        h_next = np.tanh(Z_next)

        return h_next

    def backward(self, h_next, x_t):
        """
        calculates the hidden state in the backward direction for one time step
        :param h_next: numpy.ndarray of shape (m, h)
            containing the next hidden state
        :param x_t: numpy.ndarray of shape (m, i)
            that contains the data input for the cell
            m is the batch size for the data
        :return: h_pev, the previous hidden state
        """
        # horizontal stacking of previous inner state and input
        h_x = np.concatenate((h_next, x_t), axis=1)

        # input for tanh activation
        Z_prev = np.matmul(h_x, self.Whb) + self.bhb

        # tanh activation outputting inner activated state
        h_prev = np.tanh(Z_prev)

        return h_prev

    def output(self, H):
        """
        calculates all outputs for the RNN
        :param H: numpy.ndarray of shape (t, m, 2 * h)
            that contains the concatenated hidden states from both directions,
            excluding their initialized states
            t is the number of time steps
            m is the batch size for the data
            h is the dimensionality of the hidden states
        :return: Y, the outputs
        """
        t, _, _ = H.shape

        Y = []

        for step in range(t):
            # input for softmax activation
            Z_y = np.matmul(H[step], self.Wy) + self.by

            # softmax activation
            y = softmax(Z_y)
            Y.append(y)

        Y = np.array(Y)

        return Y
