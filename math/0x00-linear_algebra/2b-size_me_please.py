#!/usr/bin/env python3
"""Contain matrix_shape function"""


def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    if matrix:
        size = [len(matrix)]
        return matrix_shape_recursion(matrix, size)
    else:
        return [0]


def matrix_shape_recursion(matrix, size):
    """recursion to calculate the shape of a matrix"""
    if type(matrix) == int:
        return size
    else:
        for element in matrix:
            new = element
            """print("current matrix {}".format(element))"""
            if type(element) != int:
                current_len = len(element)
                size.append(current_len)
            break
        return (matrix_shape_recursion(new, size))
