#!/usr/bin/env python3
"""contains the NST class"""

import numpy as np
import tensorflow as tf
import cv2


class NST:
    """
    NST class performs tasks for neural style transfer
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        constructor
        :param style_image:  image used as a style reference,
            stored as a numpy.ndarray
        :param content_image: image used as a content reference,
            stored as a numpy.ndarray
        :param alpha:
        :param beta: weight for style cost
        """
        tf.enable_eager_execution()
        if type(style_image) is not np.ndarray \
                or len(style_image.shape) != 3 \
                or style_image.shape[2] != 3:
            raise TypeError('style_image must be a'
                            'numpy.ndarray with shape (h, w, 3)')
        else:
            self.style_image = self.scale_image(style_image)

        if type(content_image) is not np.ndarray \
                or len(content_image.shape) != 3 \
                or content_image.shape[2] != 3:
            raise TypeError('content_image must be a'
                            'numpy.ndarray with shape (h, w, 3)')
        else:
            self.content_image = self.scale_image(content_image)

        if alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        else:
            self.alpha = alpha

        if beta < 0:
            raise TypeError('beta must be a non-negative number')
        else:
            self.beta = beta

    @staticmethod
    def scale_image(image):
        """

        :param image: numpy.ndarray of shape (h, w, 3)
            containing the image to be scaled
        :return:
        """
        if type(image) is not np.ndarray \
                or len(image.shape) != 3 \
                or image.shape[2] != 3:
            raise TypeError('image must be a'
                            'numpy.ndarray with shape (h, w, 3)')

        h, w, c = image.shape

        if w > h:
            w_new = 512
            h_new = int(h * 512 / w)
        else:
            h_new = 512
            w_new = int(w * 512 / h)

        # Resize the images with inter-cubic interpolation
        dim = (h_new, w_new)

        image = image[tf.newaxis, ...]
        image = tf.image.resize_images(image, dim)

        # Rescale all images to have pixel values in the range [0, 1]
        image = tf.math.divide(image, 255)

        return image
