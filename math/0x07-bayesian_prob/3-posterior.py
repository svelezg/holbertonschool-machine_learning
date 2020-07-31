#!/usr/bin/env python3
"""contains the marginal function"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data
    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param P: 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects
    :param Pr: 1D numpy.ndarray containing the prior beliefs of P
    :return: D numpy.ndarray containing the intersection of obtaining x
        and n with each probability in P
    """
    if not isinstance(n, int) or n <= 0:
        err = 'n must be a positive integer'
        raise ValueError(err)

    if not isinstance(x, int) or n < 0:
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)

    if x > n:
        err = 'x cannot be greater than n'
        raise ValueError(err)

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        err = 'P must be a 1D numpy.ndarray'
        raise TypeError(err)

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        err = 'Pr must be a numpy.ndarray with the same shape as P'
        raise TypeError(err)

    if np.any(P < 0) or np.any(P > 1):
        err = 'All values in {} must be in the range [0, 1]'.format(P)
        raise ValueError(err)

    if np.any(Pr < 0) or np.any(P > 1):
        err = 'All values in {} must be in the range [0, 1]'.format(Pr)
        raise ValueError(err)

    if not np.isclose([np.sum(Pr)], [1.])[0]:
        err = 'Pr must sum to 1'
        raise ValueError(err)

    test = np.isclose([np.sum(Pr)], [1.])

    if not test[0]:
        err = 'Pr must sum to 1'
        raise ValueError(err)

    coef = np.math.factorial(n) / \
        (np.math.factorial(x) * np.math.factorial(n - x))

    likelihood = coef * (P ** x) * (1 - P) ** (n - x)

    intersection = likelihood * Pr

    marginal = np.sum(intersection)

    result = (likelihood * Pr) / marginal

    return result
