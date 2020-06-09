#!/usr/bin/env python3
"""Contains the pool_forward function"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a convolutional layer of a neural network
    :param dA: numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the output
        of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels
    :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    :param kernel_shape: is a tuple of (kh, kw)
        containing the size of the kernel for the pooling
        containing the kernels for the convolution
        kh is the filter height
        kw is the filter width
    :param stride: tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    :param mode: string containing either max or avg,
        indicating whether to perform maximum or average pooling, respectively
    :return: partial derivatives with respect to the previous layer (dA_prev)
    """
    # Retrieving dimensions from dA
    (m, h_new, w_new, c) = dA.shape

    # Retrieving dimensions from A_prev shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieving dimensions from kernel_shape
    (kh, kw) = kernel_shape

    # Retrieving stride
    (sh, sw) = stride

    # Initialize the output with zeros
    dA_prev = np.zeros(A_prev.shape)

    for z in range(m):
        for x in range(w_new):
            for y in range(h_new):
                for v in range(c):
                    tmp_dA = dA[z, y, x, v]
                    pool = A_prev[z, y * sh: y * sh + kh,
                                  x * sw: x * sw + kw, v]

                    if mode == 'max':
                        mask = np.zeros(kernel_shape)
                        max_ = np.amax(pool)
                        np.place(mask, pool == max_, 1)
                        dA_prev[z, y * sh: y * sh + kh,
                                x * sw: x * sw + kw, v] += \
                            mask * tmp_dA
                    if mode == 'avg':
                        mask = np.ones(kernel_shape)
                        avg_ = tmp_dA / kh / kw
                        dA_prev[z,
                                y * sh: y * sh + kh,
                                x * sw: x * sw + kw: v] += \
                            mask * avg_

    return dA_prev
