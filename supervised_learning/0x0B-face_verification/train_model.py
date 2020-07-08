#!/usr/bin/env python3
"""contains the TrainModel class"""

from triplet_loss import TripletLoss
import tensorflow as tf
import numpy as np


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

    def train(self, triplets, epochs=5, batch_size=32, validation_split=0.3, verbose=True):
        """

        :param triplets: list containing the inputs to self.training_model
        :param epochs: number of epochs to train for
        :param batch_size: batch size for training
        :param validation_split: validation split for training
        :param verbose: boolean that sets the verbosity mode
        :return: History output from the training
        """
        # training
        history = self.training_model.fit(triplets,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=verbose,
                                          validation_split=validation_split)

        return history

    def save(self, save_path):
        """
        :param save_path: path to save the model
        :return: saved model
        """
        tf.keras.models.save_model(self.base_model, save_path)
        return self.base_model

    @staticmethod
    def f1_score(y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return: the f1 score
        """
        predicted = y_pred
        actual = y_true
        TP = np.count_nonzero(predicted * actual)
        TN = np.count_nonzero((predicted - 1) * (actual - 1))
        FP = np.count_nonzero(predicted * (actual - 1))
        FN = np.count_nonzero((predicted - 1) * actual)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    @staticmethod
    def accuracy(y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return: the accuracy
        """
        predicted = y_pred
        actual = y_true
        TP = np.count_nonzero(predicted * actual)
        TN = np.count_nonzero((predicted - 1) * (actual - 1))
        FP = np.count_nonzero(predicted * (actual - 1))
        FN = np.count_nonzero((predicted - 1) * actual)

        accuracy = (TP + TN) / (TP + FN + TN + FP)

        return accuracy

    def best_tau(self, images, identities, thresholds):
        """
        calculates the best tau to use for a maximal F1 score
        :param images: numpy.ndarray of shape (m, n, n, 3) containing the aligned images for testing
            m is the number of images
            n is the size of the images
        :param identities: list containing the identities of each image in images
        :param thresholds: 1D numpy.ndarray of distance thresholds (tau) to test
        :return: (tau, f1, acc)
            tau- the optimal threshold to maximize F1 score
            f1 - the maximal F1 score
            acc - the accuracy associated with the maximal F1 score
        """
        embedded = np.zeros((images.shape[0], 128))

        for i, img in enumerate(images):
            embedded[i] = self.base_model.predict(np.expand_dims(img, axis=0))[0]

        def distance(emb1, emb2):
            return np.sum(np.square(emb1 - emb2))

        distances = []  # squared L2 distance between pairs
        identical = []  # 1 if same identity, 0 otherwise

        num = len(identities)

        for i in range(num - 1):
            for j in range(i + 1, num):
                distances.append(distance(embedded[i], embedded[j]))
                identical.append(1 if identities[i] == identities[j] else 0)

        distances = np.array(distances)
        identical = np.array(identical)

        f1_scores = [self.f1_score(identical, distances < t) for t in thresholds]
        acc_scores = [self.accuracy(identical, distances < t) for t in thresholds]

        opt_idx = np.argmax(f1_scores)

        # Threshold at maximal F1 score
        opt_tau = thresholds[opt_idx]

        opt_f1 = f1_scores[opt_idx]

        # Accuracy at maximal F1 score
        opt_acc = acc_scores[opt_idx]

        return opt_tau, opt_f1, opt_acc
