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

        for l in range(1, len(layers)):
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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        :param Y: numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data
        :param A: numpy.ndarray with shape (1, m)
        containing the activated output of the neuron for each example
        :return: the cost
        """
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        :param X: a numpy.ndarray with shape (nx, m)
            that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        :param Y:  is a numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data
        :return: the neuron’s prediction and the cost of the network
        """
        A_final = self.forward_prop(X)[0]
        A_adjus = np.where(A_final >= 0.5, 1, 0)
        cost = self.cost(Y, A_final)
        return A_adjus, cost
