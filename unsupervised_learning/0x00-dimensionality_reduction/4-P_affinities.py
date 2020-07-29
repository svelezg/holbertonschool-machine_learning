#!/usr/bin/env python3
""" contains the P_affinities function"""
import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    calculates the symmetric P affinities of a data set
    :param X: numpy.ndarray of shape (n, d)
        containing the dataset to be transformed by t-SNE
        n is the number of data points
        d is the number of dimensions in each point
    :param tol: maximum tolerance allowed (inclusive) for the difference
        in Shannon entropy from perplexity for all Gaussian distributions
    :param perplexity: perplexity that all Gaussian distributions should have
    :return: P, numpy.ndarray of shape (n, n)
        containing the symmetric P affinities
    """
    n, d = X.shape

    # initialize t-SNE
    D, P, betas, H = P_init(X, perplexity)

    # traverse the pairwise distance matrix (rows)
    for i in range(n):
        # mask the row's ith element
        mask = np.ones(D[i].shape, dtype=bool)
        mask[i] = 0

        # inital row's H and P
        Hi, P[i][mask] = HP(D[i][mask], betas[i])

        # initialize limits for binary search
        high = None
        low = 0

        # binary search for row's beta to find row's H
        while abs(Hi - H) > tol:
            if Hi < H:
                high = betas[i, 0]
                betas[i, 0] = (high + low) / 2
            else:
                low = betas[i, 0]
                if high is None:
                    betas[i, 0] *= 2
                else:
                    betas[i, 0] = (high + low) / 2

            # recalculate rows H and P with updated beta
            Hi, P[i][mask] = HP(D[i][mask], betas[i])

    # make symmetric and normal
    P = (P + P.T) / (2 * n)

    return P
