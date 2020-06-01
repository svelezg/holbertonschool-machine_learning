#!/usr/bin/env python3
"""Contains the build_model function"""

import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library
    :param nx: number of input features to the network
    :param layers: list containing the number of nodes
        in each layer of the network
    :param activations: list containing the activation
        functions used for each layer of the network
    :param lambtha: L2 regularization parameter
    :param keep_prob: probability that a node will be kept for dropout
    :return: keras model
    """
    reg = tf.keras.regularizers.L1L2(l2=lambtha)

    model = tf.keras.Sequential()

    # Adds first densely-connected layer to the model:
    model.add(tf.keras.layers.Dense(units=layers[0],
                                    activation=activations[0],
                                    kernel_regularizer=reg,
                                    input_shape=(nx,),
                                    ))

    # Add subsequent densely-connected layers:
    for i in range(1, len(layers)):
        model.add(tf.keras.layers.Dropout(1 - keep_prob))
        model.add(tf.keras.layers.Dense(units=layers[i],
                                        activation=activations[i],
                                        kernel_regularizer=reg,
                                        ))

    return model
