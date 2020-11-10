#!/usr/bin/env python3
""" contains the crop_image function"""
import tensorflow as tf


def crop_image(image, size):
    """
    performs a random crop of an image
    :param image: 3D tf.Tensor containing the image to crop
    :param size: tuple containing the size of the crop
    :return: the cropped image
    """
    img = tf.random_crop(image, size=size)
    return img
