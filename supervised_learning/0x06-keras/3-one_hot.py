#!/usr/bin/env python3
"""Contains the one_hot function"""

import tensorflow as tf


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix
    :param labels:
    :param classes:
    :return: the one-hot matrix
    """
    return tf.keras.utils.to_categorical(
        labels, classes
    )
