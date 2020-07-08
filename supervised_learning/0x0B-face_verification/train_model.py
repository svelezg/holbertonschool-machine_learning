#!/usr/bin/env python3
"""contains the TrainModel class"""

from triplet_loss import TripletLoss
import tensorflow as tf


class TrainModel:
    """
    TrainModel class
    """

    def __init__(self, model_path, alpha):
        """
        constructor
        :param model_path: path to the base face verification embedding model
            loads the model using with tf.keras.utils.CustomObjectScope({'tf': tf}):
            saves this model as the public instance method base_model
        :param alpha: alpha to use for the triplet loss calculation
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = tf.keras.models.load_model(model_path)

        A = tf.keras.Input(shape=(96, 96, 3))
        P = tf.keras.Input(shape=(96, 96, 3))
        N = tf.keras.Input(shape=(96, 96, 3))

        network0 = self.base_model(A)
        network1 = self.base_model(P)
        network2 = self.base_model(N)

        tl = TripletLoss(alpha)

        # combine the output of the three branches
        combined = [network0, network1, network2]
        output = tl(combined)

        my_model = tf.keras.models.Model([A, P, N], output)

        my_model.compile(optimizer='adam')

        self.training_model = my_model
