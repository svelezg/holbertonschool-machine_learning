#!/usr/bin/env python3
"""Contain add_matrices2D function"""


def add_matrices2D(arr1, arr2):
    """ adds two matrices element-wise"""

    if len(arr1) != len(arr2):
        return None
    else:
        new_matrix = []

        for i in range(len(arr1)):
            new_row = []
            if len(arr1[i]) == len(arr2[i]):
                for j in range(len(arr1)):
                    new_row.append(arr1[i][j] + arr2[i][j])
                new_matrix.append(new_row)
            else:
                return None
        return new_matrix
