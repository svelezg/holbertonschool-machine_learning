#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 9))

plt.subplot(221)
plt.bar(names, values)

plt.subplot(222)
plt.scatter(names, values)

plt.subplot(212)
plt.plot(names, values)

plt.suptitle('Categorical Plotting')
plt.show()
