#!/usr/bin/env python3
"""Contain add_matrices2D function"""


def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    if matrix:
        size = [len(matrix)]
        return matrix_shape_recursion(matrix, size)
    else:
        return 0


def matrix_shape_recursion(matrix, size):
    """recursion to calculates the shape of a matrix"""
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


def add_matrices2D(arr1, arr2):
    """ adds two matrices element-wise"""

    shape1 = matrix_shape(arr1)
    shape2 = matrix_shape(arr2)


    if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
        return None
    else:
        new_matrix = []

        for i in range(shape1[0]):
            new_row = []
            for j in range (shape1[1]):
                new_row.append(arr1[i][j] + arr2[i][j])
            new_matrix.append(new_row)
        return new_matrix
