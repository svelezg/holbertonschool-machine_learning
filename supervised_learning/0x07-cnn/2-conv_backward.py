#!/usr/bin/env python3
"""Contains the conv_backward function"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network
    :param dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
        containing the partial derivatives with respect to the unactivated
        output of the convolutional layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
    :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    :param W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        containing the kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
    :param b: numpy.ndarray of shape (1, 1, 1, c_new)
        containing the biases applied to the convolution
    :param padding: string that is either same or valid,
        indicating the type of padding used
    :param stride: tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    :return: the partial derivatives with respect to
        the previous layer (dA_prev),
        the kernels (dW),
        and the biases (db), respectively
    """
    # Retrieving dimensions from dZ
    m, h_new, w_new, c_new = dZ.shape

    # Retrieving dimensions from A_prev shape
    _, h_prev, w_prev, c_prev = A_prev.shape

    # Retrieving dimensions from W's shape
    kh, kw, _, _ = W.shape

    # Retrieving stride
    sh, sw = stride

    # Setting padding for valid
    pw, ph = 0, 0

    # bias calculation
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Setting padding for same
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))

    # pad images
    A_prev = np.pad(A_prev,
                    pad_width=((0, 0),
                               (ph, ph),
                               (pw, pw),
                               (0, 0)),
                    mode='constant', constant_values=0)

    # Initializing dX, dW with the correct shapes
    dW = np.zeros_like(W)
    dA_prev = np.zeros_like(A_prev)

    # Looping over vertical(h) and horizontal(w) axis of the output
    for z in range(m):
        for y in range(h_new):
            for x in range(w_new):
                # over every channel
                for v in range(c_new):
                    tmp_W = W[:, :, :, v]
                    tmp_dz = dZ[z, y, x, v]

                    dA_prev[z, y * sh: y * sh + kh,
                            x * sw: x * sw + kw,
                            :] += tmp_dz * tmp_W

                    tmp_A_prev = A_prev[z, y * sh: y * sh + kh,
                                        x * sw: x * sw + kw,
                                        :]
                    dW[:, :, :, v] += tmp_A_prev * tmp_dz

    # subtracting padding
    dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
