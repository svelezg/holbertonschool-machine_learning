#!/usr/bin/env python3
"""contains the FaceVerification class"""

import numpy as np
import tensorflow as tf


class FaceVerification:
    """
    FaceVerification class
    """

    def __init__(self, model, database, identities):
        """
        constructor
        :param model: either the face verification embedding model
            or the path to where the model is stored
        :param database: numpy.ndarray of all the face embeddings
            in the database
        :param identities: list of identities corresponding
            to the embeddings in the database
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.model = tf.keras.models.load_model(model)

        self.database = database
        self.identities = identities

    def embedding(self, images):
        """
        calculates the face embedding of images
        :param images: numpy.ndarray of shape (i, n, n, 3)
            containing the aligned images
            i is the number of images
            n is the size of the aligned images
        :return: numpy.ndarray of shape (i, e) containing
            the embeddings where e is the dimensionality of the embeddings
        """
        embedded = np.zeros((images.shape[0], 128))

        for i, img in enumerate(images):
            embedded[i] = self.model.predict(np.expand_dims(img, axis=0))[0]

        return np.array(embedded)

    def verify(self, image, tau=0.5):
        """
        :param image: numpy.ndarray of shape (n, n, 3)
            containing the aligned image of the face to be verify
        :param tau: the maximum euclidean distance used for verification
        :return: (identity, distance), or (None, None) on failure
            identity is the identity of the verified face
            distance is the euclidean distance between the verified
            face embedding and the identified database embedding
        """

        my_embedding = self.model.predict(np.expand_dims(image, axis=0))[0]

        def distance(emb1, emb2):
            return np.sum(np.square(emb1 - emb2))

        distances = []  # squared L2 distance between pairs

        num = len(self.identities)

        for i in range(num):
            distances.append(distance(my_embedding, self.database[i]))

        distances = np.array(distances)

        idx = np.argmin(distances)

        if distances[idx] < tau:
            return self.identities[idx], distances[idx]
        else:
            return None, None
