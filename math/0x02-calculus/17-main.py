#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))

print('*********')

poly = []
print(poly_integral(poly, 3))

print('*********')

poly = [5, 3, 0, 0, 0]
print(poly_integral(poly))
