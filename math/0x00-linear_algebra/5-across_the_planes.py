#!/usr/bin/env python3
"""Contain add_matrices2D function"""


def add_matrices2D(mat1, mat2):
    """ adds two matrices element-wise"""

    if len(mat1) == len(mat2):
        new_matrix = []
        for i in range(len(mat1)):
            new_row = []
            if len(mat1[i]) == len(mat2[i]):
                for j in range(len(mat1[i])):
                    new_row.append(mat1[i][j] + mat2[i][j])
                new_matrix.append(new_row)
            else:
                return None
        return new_matrix
