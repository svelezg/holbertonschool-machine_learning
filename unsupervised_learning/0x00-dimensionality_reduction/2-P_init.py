#!/usr/bin/env python3
""" contains the P_init function"""
import numpy as np


def P_init(X, perplexity):
    """
    initializes all variables required to calculate the P affinities in t-SNE
    :param X: numpy.ndarray of shape (n, d)
        containing the dataset to be transformed by t-SNE
        n is the number of data points
        d is the number of dimensions in each point
    :param perplexity: perplexity that all Gaussian distributions should have
    :return: (D, P, betas, H)
        D: a numpy.ndarray of shape (n, n)
            that calculates the pairwise distance between two data points
        P: a numpy.ndarray of shape (n, n)
            initialized to all 0‘s that will contain the P affinities
        betas: a numpy.ndarray of shape (n, 1)
            initialized to all 1’s that will contain all of the beta values
        H is the Shannon entropy for perplexity perplexity
    """
    n, d = X.shape

    # (a - b)**2 = a^2 - 2ab + b^2 expansion
    a2 = np.sum(X ** 2, axis=1)
    b2 = np.sum(X ** 2, axis=1)[:, np.newaxis]
    ab = np.matmul(X, X.T)
    D = a2 - 2 * ab + b2

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, betas, H
