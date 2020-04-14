#!/usr/bin/env python3
"""Contain matrix_shape function"""


def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    if matrix:
        shape = [len(matrix)]
        while type(matrix[0]) == list:
            shape.append(len(matrix[0]))
            matrix = matrix[0]
        return shape
    else:
        return [0]
