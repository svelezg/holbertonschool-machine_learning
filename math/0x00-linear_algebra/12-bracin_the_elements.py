#!/usr/bin/env python3
"""Contains the np_elementwise function"""
import numpy as np


def np_elementwise(mat1, mat2):
    """performs element-wise addition, subtraction,
    multiplication, and division"""
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
