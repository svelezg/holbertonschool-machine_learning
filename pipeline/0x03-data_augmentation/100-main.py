#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os

pca_color = __import__('100-pca').pca_color

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(100)
np.random.seed(100)
doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    alphas = np.random.normal(0, 0.1, 3)
    fig = plt.figure(figsize=(4, 4), facecolor="w")
    plt.imshow(pca_color(image, alphas))
    plt.show()

    # save as png
    outfile = os.path.join('images', '100-main' + ".png")

    fig.savefig(outfile, dpi=150)
