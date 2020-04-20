#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

data = {'a': np.arange(50),
        'c': np.random.randint(1000, 5000, 50),
        'd': np.random.randn(50)}

data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 1000

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()
