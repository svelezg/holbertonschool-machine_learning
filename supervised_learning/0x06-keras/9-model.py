#!/usr/bin/env python3
"""Contains the save_model and the load_model function"""

import tensorflow as tf
import matplotlib.pyplot as plt


def save_model(network, filename):
    """
    saves an entire model
    :param network: model to save
    :param filename: path of the file that the model should be saved to
    :return: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    loads an entire model
    :param filename:  path of the file that the model should be loaded from
    :return: loaded model
    """
    network = tf.keras.models.load_model(filename)
    return network
