# !/usr/bin/env python3

import numpy as np

specificity = __import__('3-specificity').specificity
specificity1 = __import__('3b-specificity').specificity
if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(specificity(confusion))
    print(specificity1(confusion))