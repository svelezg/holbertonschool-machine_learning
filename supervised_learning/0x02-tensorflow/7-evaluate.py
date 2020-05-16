#!/usr/bin/env python3
"""Contains the train function"""

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders


def evaluate(X, Y, save_path):
    """
    evaluates the output of a neural network:
    :param X: numpy.ndarray containing the input data to evaluate
    :param Y: numpy.ndarray containing the one-hot labels for X
    :param save_path: location to load the model from
    :return: the network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)

        # collect x and y placeholders
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        # define feed_dict my_input with X as x and Y as y
        my_input = {x: X, y: Y}

        # collect y_pred, result of the forward_prop operation
        y_pred = tf.get_collection("y_pred")[0]
        # execute the forward_prop operation with X and Y as input
        prediction_ = sess.run(y_pred, feed_dict=my_input)

        # collect accuracy, result of the calculate_accuracy operation
        accuracy = tf.get_collection("accuracy")[0]
        # execute the calculate_accuracy operation with X and Y as input
        accuracy_ = sess.run(accuracy, feed_dict=my_input)

        # collect loss, result of the calculate_loss operation
        loss = tf.get_collection("loss")[0]
        # execute the calculate_loss operation with X and Y as input
        # to get the _cost
        cost_ = sess.run(loss, feed_dict=my_input)

    return prediction_, accuracy_, cost_
