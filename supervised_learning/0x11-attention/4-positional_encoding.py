#!/usr/bin/env python3
"""contains the positional_encoding function"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    calculates the positional encoding for a transformer
    :param max_seq_len: integer representing the maximum sequence length
    :param dm: model depth
    :return: numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encoding vectors
    """
    # position
    t = np.arange(max_seq_len)[:, np.newaxis]

    # model depth
    index = np.arange(dm)[np.newaxis, :]
    dm_float = np.float32(dm)

    # angle
    W = 1 / (np.power(10000, (2*(index//2)/dm_float)))

    # argument
    Wt = (W * t)

    positional_vect = np.zeros((max_seq_len, dm))

    # sin to even indices
    positional_vect[:, 0::2] = np.sin(Wt[:, 0::2])

    # cos to odd indices
    positional_vect[:, 1::2] = np.cos(Wt[:, 1::2])

    return positional_vect
