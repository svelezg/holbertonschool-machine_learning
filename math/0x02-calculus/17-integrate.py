#!/usr/bin/env python3
"""Contains the function poly_integral(poly, C=0)"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    "0. C checks"
    if not isinstance(C, (int, float)):
        return None

    "1. poly checks"
    if type(poly) != list:
        return None
    if len(poly) == 0:
        return None
    if poly == [0]:
        return [C]
    for element in poly:
        if not isinstance(element, (int, float)):
            return None

    "2. iteration"
    deg = len(poly)
    integral = [C]
    for i in range(deg):
        coef_integral = poly[i] / (i + 1)
        if coef_integral % 1 == 0:
            coef_integral = int(coef_integral)
        integral.append(coef_integral)
    index = len(integral) - 1

    "3. Trim zeros at the end"
    while integral[index] == 0:
        integral.pop(index)
        i -= 1

    "4. Return Trimmed list"
    return integral
