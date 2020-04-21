#!/usr/bin/env python3
"""Contains the function poly_derivative(poly)"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    deg = len(poly)
    if deg <= 0:
        return None

    if deg == 1:
        return [0]

    derivative = []

    for i in range(1, deg):
        conf_derivative = poly[i] * i
        derivative.append(conf_derivative)

    return derivative
