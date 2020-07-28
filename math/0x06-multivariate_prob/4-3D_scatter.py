#!/usr/bin/env python3

import numpy as np
from multinormal import MultiNormal
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

if __name__ == '__main__':

    np.random.seed(0)

    data = np.random.multivariate_normal([12, 10],
                                         [[12, -10],
                                          [-10, 16]],
                                         10000).T

    mn = MultiNormal(data)

    prob = []
    for elem in data.T:
        elem = np.expand_dims(elem, axis=1)
        res = mn.pdf(elem)
        prob.append(res)

    # plot
    fig = plt.figure(figsize=(4, 4), facecolor="w")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], prob, c=prob, cmap=cm.coolwarm, s=1)

    """
    # Customize the z axis.
    ax.set_zlim(0, 0.003)
    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
    """


    plt.show()

    fig = plt.figure(figsize=(4, 4), facecolor="w")
    ax = fig.add_subplot(111)
    ax.scatter(data[0], data[1], c=prob, cmap=cm.coolwarm, s=1)
    plt.show()
