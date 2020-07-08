#!/usr/bin/env python3
"""contains the FaceVerification class"""

import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
import dlib
import tensorflow as tf
import tensorflow.keras as K


class FaceVerification:
    """
    FaceVerification class
    """

    def __init__(self, model, database, identities):
        """

        :param model: either the face verification embedding model or the path to where the model is stored
        :param database: numpy.ndarray of all the face embeddings in the database
        :param identities: list of identities corresponding to the embeddings in the database
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.model = tf.keras.models.load_model(model)

        self.database = database
        self.identities = identities

    def embedding(self, images):
        """

        :param images: the images to retrieve the embeddings of
        :return: numpy.ndarray of embeddings
        """

    def verify(self, image, tau=0.5):
        """

        :param image: the aligned image of the face to be verify
        :param tau: the maximum euclidean distance used for verification
        :return: (identity, distance), or (None, None) on failure
            identity is the identity of the verified face
            distance is the euclidean distance between the verified face embedding and the identified database embedding
        """