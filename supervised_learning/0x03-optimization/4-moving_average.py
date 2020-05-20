#!/usr/bin/env python3
"""Contains the  moving_average function"""


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set
    :param data: list of data to calculate the moving average of
    :param beta: weight used for the moving average
    :return: list containing the moving averages of data
    """
    avg_list = []
    value = 0
    for i in range(len(data)):
        # Exponentially Weighted Averages
        value = beta * value + (1 - beta) * (data[i])

        # Bias Correction of Exponentially Weighted Averages
        correction = 1 / (1 - beta ** (i + 1))

        avg_list.append(correction * value)

    return avg_list
