#!/usr/bin/env python3
"""contains the sdp_attention function"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    calculates the scaled dot product attention
    :param Q: tensor with its last two dimensions as (..., seq_len_q, dk)
        containing the query matrix
    :param K: tensor with its last two dimensions as (..., seq_len_v, dk)
        containing the key matrix
    :param V: tensor with its last two dimensions as (..., seq_len_v, dv)
        containing the value matrix
    :param mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
        containing the optional mask, or defaulted to None
    :return: output, weights
        outputa tensor with its last two dimensions as
            (..., seq_len_q, dv)
            containing the scaled dot product attention
        weights a tensor with its last two dimensions as
            (..., seq_len_q, seq_len_v)
            containing the attention weights
    """
    dk = tf.shape(Q)[-1]
    dk_float = tf.cast(dk, tf.float32)

    scaled = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk_float)

    if mask is not None:
        scaled += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
