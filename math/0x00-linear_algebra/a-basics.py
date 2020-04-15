#!/usr/bin/env python3
import numpy as np

a = np.arange(15).reshape(3, 5)
print("a: {}".format(a))

print("shape: {}".format(a.shape))

print("ndim: {}".format(a.ndim))

print("dtype: {}".format(a.dtype.name))

print("item size: {}".format(a.itemsize))

print("size: {}".format(a.size))

print("type(a): {}".format(type(a)))


b = np.array([6, 7, 8])

print("b: {}".format(b))

print("type(b): {}".format(type(b)))

b = np.array([1.2, 3.5, 5.1])
print("b: {}".format(b))
print("dtype b: {}".format(b.dtype.name))
