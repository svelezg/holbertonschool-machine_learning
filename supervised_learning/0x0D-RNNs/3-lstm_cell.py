#!/usr/bin/env python3
"""contains the LSTMCell class"""

import numpy as np


def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """softmax function"""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class LSTMCell:
    """
    class LSTMCell that represents a represents an LSTM unit
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
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        # Biases of the cell
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        performs forward propagation for one time step
        :param h_prev: h_prev: numpy.ndarray of shape (m, h)
            containing the previous hidden state
        :param c_prev: numpy.ndarray of shape (m, h)
            containing the previous cell state
        :param x_t: numpy.ndarray of shape (m, i)
            that contains the data input for the cell
        :return: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """

        # horizontal stacking of previous inner state and input
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # 1. Forget gate (f)
        f_t = sigmoid(np.matmul(h_x, self.Wf) + self.bf)

        # 2. Update gate (r)
        u_t = sigmoid(np.matmul(h_x, self.Wu) + self.bu)
        C_t_tilde = np.tanh(np.matmul(h_x, self.Wc) + self.bc)

        c_next = f_t * c_prev + u_t * C_t_tilde

        # 3. Output gate (0)
        o_t = sigmoid(np.matmul(h_x, self.Wo) + self.bo)
        h_next = o_t * np.tanh(c_next)

        # input for softmax activation
        Z_y = np.matmul(h_next, self.Wy) + self.by

        # softmax activation outputting LSTM final output
        y = softmax(Z_y)

        return h_next, c_next, y
