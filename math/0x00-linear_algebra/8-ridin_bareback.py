#!/usr/bin/env python3
"""Contains the mat_mul function"""


def mat_mul(mat1, mat2):
    """performs matrix multiplication"""

    if len(mat1[0]) != len(mat2):
        return None

    members = len(mat1[0])
    rows = len(mat1)
    columns = len(mat2[0])

    matrix = []
    for i in range(rows):
        row = []
        for j in range(columns):
            element = 0
            for k in range(members):
                a = mat1[i][k] * mat2[k][j]
                element = element + a
            row.append(element)
        matrix.append(row)

    return matrix
