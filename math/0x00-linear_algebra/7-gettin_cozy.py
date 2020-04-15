#!/usr/bin/env python3
"""Contains the cat_matrices2D function"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis:"""
    matrix = []

    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None

    if axis == 1 and len(mat1) != len(mat2):
        return None

    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j])
        if axis == 1:
            for j in range(len(mat2[0])):
                row.append(mat2[i][j])
        matrix.append(row)

    if axis == 0:
        for i in range(len(mat2)):
            row = []
            for j in range(len(mat2[0])):
                row.append(mat2[i][j])
            matrix.append(row)

    return matrix
