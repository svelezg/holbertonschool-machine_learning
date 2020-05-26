#!/usr/bin/env python3
"""Contains the specificity function"""

import numpy as np


def specificity(confusion):
    """

    :param confusion: is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct
    labels and column indices represent the predicted labels
    :return: numpy.ndarray of shape (classes,)
    containing the specificity of each class
    """
    predicted = confusion.sum(axis=0)
    true_positives = confusion.diagonal()
    false_positives = predicted - true_positives

    all_positives = true_positives.sum()
    true_negatives = all_positives - true_positives

    return true_negatives / (true_negatives + false_positives)
