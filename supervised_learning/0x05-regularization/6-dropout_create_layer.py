#!/usr/bin/env python3
"""Contains the dropout_create_layer"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout
    :param prev: tensor containing the output of the previous layer
    :param n: number of nodes the new layer should contain
    :param activation: activation func that should be used on the layer
    :param keep_prob: probability that a node will be kept
    :return: output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init)
    dropout = tf.layers.Dropout(keep_prob)
    return dropout(model(prev))
