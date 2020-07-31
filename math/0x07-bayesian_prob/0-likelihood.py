#!/usr/bin/env python3
"""contains the likelihood function"""

import numpy as np


def likelihood(x, n, P):
    """
    calculates the likelihood of obtaining this data
    given various hypothetical probabilities of
    developing severe side effects
    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param P: 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects
    :return: 1D numpy.ndarray containing the likelihood of obtaining the data,
        x and n, for each probability in P
    """
    if not isinstance(n, int) or n <= 0:
        err = 'n must be a positive integer'
        raise ValueError(err)

    if not isinstance(x, int) or x < 0:
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)

    if x > n:
        err = 'x cannot be greater than n'
        raise ValueError(err)

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        err = 'P must be a 1D numpy.ndarray'
        raise TypeError(err)

    if np.any(P < 0) or np.any(P > 1):
        err = 'All values in P must be in the range [0, 1]'
        raise ValueError(err)

    coef = np.math.factorial(n) / \
        (np.math.factorial(x) * np.math.factorial(n - x))

    result = coef * (P ** x) * (1 - P) ** (n - x)

    return result
