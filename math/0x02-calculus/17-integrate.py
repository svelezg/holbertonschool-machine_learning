#!/usr/bin/env python3
"""Contains the function poly_integral(poly, C=0)"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    deg = len(poly)

    integral = [C]

    for i in range(0, deg):
        coef_integral = poly[i] / (i + 1)
        if coef_integral % 1 == 0:
            coef_integral = int(coef_integral)
        integral.append(coef_integral)

    return integral
