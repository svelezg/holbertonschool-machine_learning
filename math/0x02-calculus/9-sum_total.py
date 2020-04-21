#!/usr/bin/env python3
"""Contain the summation_i_squared(n) function"""


def summation_i_squared(n):
    """calculates the summation i squared"""
    result = n * (1/6) + (n ** 2) * (1/2) + (n ** 3) * (1/3)
    return result
