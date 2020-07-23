#!/usr/bin/env python3

import numpy as np


def definiteness(matrix):
    """
    calculates the definiteness of a matrix:
    :param matrix: numpy.ndarray of shape (n, n)
        whose definiteness should be calculated
    :return:
    """
    # type test
    err = 'matrix must be a numpy.ndarray'
    if not isinstance(matrix, np.ndarray):
        raise TypeError(err)

    # square test
    my_len = matrix.shape[0]
    if len(matrix.shape) != 2 or my_len != matrix.shape[1]:
        return None

    # symmetry test
    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None

    # list of sub matrices (up-left to down-right)
    sub_matrices = [matrix[:i, :i] for i in range(1, my_len + 1)]

    # eigenvalues
    w, v = np.linalg.eig(matrix)

    if all(w > 0):
        return '(Positive definite) all eigenvalues positive'
    elif all(w >= 0):
        return '(Positive semi-definite) all eigenvalues non-negative'
    elif all(w < 0):
        return '(Negative definite) all eigenvalues negative'
    elif all(w <= 0):
        return '(Negative semi-definite) all eigenvalues non-positive'
    else:
        return '(Indefinite) both positive and negative eigenvalues'
