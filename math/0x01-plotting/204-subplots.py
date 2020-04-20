#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

plt.rc('axes', labelsize='x-small')

# Create a figure
fig = plt.figure()


axes = fig.subplots(3, 2)
axes[0, 0].plot(x, y, 'r--')
axes[0, 1].plot(x, y)
axes[1, 0].scatter(x, y)
axes[1, 1].scatter(x, y, s=7, c='m')

gs = axes[2, 0].get_gridspec()
print(gs)
for ax in axes[2, 0:]:
    ax.remove()
axbig = fig.add_subplot(gs[2, 0:])

axbig.plot(x, y, 'b')

plt.suptitle('All in One')
plt.show()
