#!/usr/bin/env python3
""" contains the tsne function"""
import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
     performs a t-SNE transformation:
    :param X: numpy.ndarray of shape (n, d)
        containing the dataset to be transformed by t-SNE
        n is the number of data points
        d is the number of dimensions in each point
    :param ndims: new dimensional representation of X
    :param idims: intermediate dimensional representation of X after PCA
    :param perplexity: the perplexity
    :param iterations:  number of iterations
    :param lr: learning rate
    :return: Y, a numpy.ndarray of shape (n, ndim)
        containing the optimized low dimensional transformation of X
    """
    X = pca(X, idims)

    n, d = X.shape

    # early exaggeration with an exaggeration of 4
    P = P_affinities(X, perplexity=perplexity) * 4

    Y = np.random.randn(n, ndims)
    Yprev = Y

    for i in range(0, iterations):
        if i != 0 and i % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i, C))

        dY, Q = grads(Y, P)

        # momentum
        if i <= 20:
            momentum = 0.5
        else:
            momentum = 0.8

        # update
        tmp = Y

        Y = Y - lr * dY + momentum * (Y - Yprev)
        Yprev = tmp
        Y = Y - np.mean(Y, axis=0)

        if i == 100:
            P = P / 4.

    return Y
