#!/usr/bin/env python3
"""Contains the DeepNeuralNetwork class"""

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=200, edgeitems=8, linewidth=55)


class DeepNeuralNetwork:
    """
    DeepNeuralNetwork class
    defines a deep neural network
    performing binary classification:
    """

    def __init__(self, nx, layers):
        """
        Class constructor
        :param nx: the number of input features
        :param layers: list representing the number of nodes
        in each layer of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        for nodes in layers:
            if nodes < 1 or not isinstance(nodes, int):
                raise ValueError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        """The weights vector for the hidden layer.
            Upon instantiation, it should be initialized
            using a random normal distribution"""
        self.weights['W1'] = np.random.randn(layers[0], nx) * np.sqrt(2/nx)

        """The bias for the hidden layer. Upon instantiation,
            it should be initialized with 0’s"""
        self.weights['b1'] = np.zeros((layers[0], 1))

        for l in range(1, len(layers)):
            W_key = "W{}".format(l + 1)
            b_key = "b{}".format(l + 1)
            self.weights[W_key] = np.random.randn(layers[l], layers[l - 1]) \
                * np.sqrt(2 / layers[l - 1])
            self.weights[b_key] = np.zeros((layers[l], 1))
