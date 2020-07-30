#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
tsne = __import__('8-tsne').tsne

np.random.seed(0)
X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
Y = tsne(X, perplexity=30.0, iterations=1200, lr=750)

fig = plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], 20, labels, cmap=cm.coolwarm)
plt.colorbar()
plt.title('t-SNE')
plt.show()
fig.savefig('t-SNE.png')
