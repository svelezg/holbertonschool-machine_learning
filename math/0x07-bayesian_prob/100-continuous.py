#!/usr/bin/env python3
"""contains the posterior function"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    calculates the posterior probability that the probability of
    developing severe side effects falls within a specific range
    given the data
    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param p1: lower bound on the range
    :param p2: upper bound on the range
    :return: posterior probability that p is within the range
        [p1, p2] given x and n
    """
    if not isinstance(n, int) or n < 1:
        err = 'n must be a positive integer'
        raise ValueError(err)

    if not isinstance(x, int) or x < 0:
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)

    if x > n:
        err = 'x cannot be greater than n'
        raise ValueError(err)

    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        err = 'p1 must be a float in the range [0, 1]'
        raise ValueError(err)

    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        err = 'p2 must be a float in the range [0, 1]'
        raise ValueError(err)

    if p2 <= p1:
        err = 'p2 must be greater than p1'
        raise ValueError(err)

    a = x + 1
    b = n - x + 1

    # Cumulative distribution function
    cum_beta1 = special.btdtr(a, b, p1)
    cum_beta2 = special.btdtr(a, b, p2)

    result = cum_beta2 - cum_beta1

    return result
