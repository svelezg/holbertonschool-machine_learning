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
        c_new is the number of channels
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
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)
    for m_i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c_i in range(c):
                    pool = A_prev[m_i, h * sh:(kh + h * sh), w * sw:(kw + w * sw), c_i]
                    dA_val = dA[m_i, h, w, c_i]
                    if mode == 'max':
                        zero_mask = np.zeros(kernel_shape)
                        _max = np.amax(pool)
                        np.place(zero_mask, pool == _max, 1)
                        dA_prev[m_i, h * sh:(kh + h * sh),
                        w * sw:(kw + w * sw), c_i] += zero_mask * dA_val
                    if mode == 'avg':
                        avg = dA_val / kh / kw
                        one_mask = np.ones(kernel_shape)
                        dA_prev[m_i, h * sh:(kh + h * sh),
                        w * sw:(kw + w * sw), c_i] += one_mask * avg
    return dA_prev
