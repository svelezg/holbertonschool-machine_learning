#!/usr/bin/env python3
"""Contains the cat_arrays function"""


def cat_arrays(arr1, arr2):
    """concatenates two arrays"""
    arr = arr1.copy()
    for i in range(len(arr2)):
        arr.append(arr2[i])
    return arr
