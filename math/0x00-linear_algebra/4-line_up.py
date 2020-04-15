#!/usr/bin/env python3
"""Contain add_arrays function"""


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""

    shape1 = len(arr1)
    shape2 = len(arr2)

    if shape1 != shape2:
        return None
    else:
        new_array = []

        for i in range(shape1):
            new_array.append(arr1[i] + arr2[i])
        return new_array
