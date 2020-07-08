#!/usr/bin/env python3
"""contains the TripletLoss class"""

import tensorflow as tf
import tensorflow.keras as K


class TripletLoss(K.layers.Layer):
    """
    TripletLoss class
    """

    def __init__(self, alpha, **kwargs):
        """

        :param alpha: alpha value used to calculate the triplet loss
        :param kwargs:
        sets the public instance attribute alpha
        """
        super(TripletLoss, self).__init__(**kwargs)
        self.alpha = alpha

    def triplet_loss(self, inputs):
        """
        :param inputs: list containing the anchor,
        positive and negative output
            tensors from the last layer of the model,
        :return: tensor containing the triplet loss values
        """
        A, P, N = inputs

        diff1 = K.layers.Subtract()([A, P])
        diff2 = K.layers.Subtract()([A, N])

        diff1_sqr = K.backend.square(diff1)
        diff2_sqr = K.backend.square(diff2)

        diff1_sum = K.backend.sum(diff1_sqr, axis=1)
        diff2_sum = K.backend.sum(diff2_sqr, axis=1)

        diff_sum = K.layers.Subtract()([diff1_sum, diff2_sum])

        loss = K.backend.maximum(diff_sum + self.alpha, 0)
        return loss

    def call(self, inputs):
        """
        adds the triplet loss to the graph
        :param inputs: list containing the anchor,
        positive, and negative output tensors from the last layer of the model
        :return: the triplet loss tensor
        """
        triplet_loss_tensor = self.triplet_loss(inputs)
        self.add_loss(triplet_loss_tensor)
        return triplet_loss_tensor
