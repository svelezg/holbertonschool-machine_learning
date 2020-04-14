#!/usr/bin/env python3
"""Contain matrix_transpose function"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix"""
    columns = len(matrix)
    rows = len(matrix[0])

    new_matrix = []
    for i in range(rows):
        new_row = []
        for j in range(columns):
            new_row.append(matrix[j][i])
        new_matrix.append(new_row)
    return new_matrix
