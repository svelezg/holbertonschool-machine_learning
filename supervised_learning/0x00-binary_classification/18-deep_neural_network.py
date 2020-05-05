#!/usr/bin/env python3
"""Contains the DeepNeuralNetwork class"""

import numpy as np


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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        """The weights vector for the hidden layer.
            Upon instantiation, it should be initialized
            using a random normal distribution"""
        self.__weights['W1'] = np.random.randn(layers[0], nx) * np.sqrt(2/nx)

        """The bias for the hidden layer. Upon instantiation,
            it should be initialized with 0’s"""
        self.__weights['b1'] = np.zeros((layers[0], 1))

        for l in range(1, self.__L):
            W_key = "W{}".format(l + 1)
            b_key = "b{}".format(l + 1)
            self.__weights[W_key] = np.random.randn(layers[l], layers[l - 1]) \
                * np.sqrt(2 / layers[l - 1])
            self.__weights[b_key] = np.zeros((layers[l], 1))

    @property
    def L(self):
        """property to retrieve L"""
        return self.__L

    @property
    def cache(self):
        """property to retrieve b1"""
        return self.__cache

    @property
    def weights(self):
        """property to retrieve A1"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        :param X: a numpy.ndarray with shape (nx, m)
        that contains the input data
        :return: the output of the neural network and the cache,
        respectively
        """
        self.__cache['A0'] = X

        for l in range(self.__L):
            W_key = "W{}".format(l + 1)
            b_key = "b{}".format(l + 1)
            A_key_prev = "A{}".format(l)
            A_key_forw = "A{}".format(l + 1)

            Z = np.matmul(self.__weights[W_key], self.__cache[A_key_prev]) \
                + self.__weights[b_key]
            self.__cache[A_key_forw] = 1 / (1 + np.exp(-Z))

        return self.__cache[A_key_forw], self.__cache
