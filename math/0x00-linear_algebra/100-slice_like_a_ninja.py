#!/usr/bin/env python3
"""Contains the np_slice function"""


def np_slice(matrix, axes={}):
    """slices a matrix along a specific axes"""

    empty_slicer = slice(None, None, None)
    slice_list = [empty_slicer] * len(matrix.shape)

    for key, value in sorted(axes.items()):
        print(*value)
        slice_list[key] = slice(*value)
    matrix = matrix[tuple(slice_list)]
    return matrix
