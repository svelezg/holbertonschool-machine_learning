#!/usr/bin/env python3
"""contains the SelfAttention class"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """attention for machine translation"""

    def __init__(self, units):
        """
        Class constructor
        :param units: integer representing the number of hidden units
            in the alignment model
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        :param s_prev: tensor of shape (batch, units)
            containing the previous decoder hidden state
        :param hidden_states: tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder
        :return: context, weights
            context is a tensor of shape (batch, units)
            that contains the context vector for the decoder
            weights is a tensor of shape (batch, input_seq_len, 1)
            that contains the attention weights
        """
        # expansion to match hidden_states dimensions (input_seq_len)
        exp_s_prev = tf.expand_dims(s_prev, axis=1)

        # weights calculation
        score = self.V(tf.nn.tanh(self.W(exp_s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)

        # context as the weighted sum of the hidden_states
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
