#!/usr/bin/env python3
"""Contains the lenet5 function"""

import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5 architecture using tensorflow
    :param x: tf.placeholder of shape (m, 28, 28, 1)
        containing the input images for the network
        m is the number of images
    :param y: tf.placeholder of shape (m, 10)
        containing the one-hot labels for the network
    :return:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
            (with default hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    # implement He et. al initialization for the layer weights
    initializer = \
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Conv layers
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    model0 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              name='layer')
    layer0 = model0(x)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    model1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2))
    layer1 = model1(layer0)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    model2 = tf.layers.Conv2D(filters=16,
                              kernel_size=(5, 5),
                              padding='valid',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              name='layer')
    layer2 = model2(layer1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    model3 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2))
    layer3 = model3(layer2)

    # Flattening between conv and dense layers
    model_flatt = tf.layers.Flatten
    layer3 = model_flatt()(layer3)

    # Fully connected (Dense) layers
    # Fully connected layer with 120 nodes
    model4 = tf.layers.Dense(units=120,
                             activation=tf.nn.relu,
                             kernel_initializer=initializer,
                             name='layer')
    layer4 = model4(layer3)

    # Fully connected layer with 84 nodes
    model5 = tf.layers.Dense(units=84,
                             activation=tf.nn.relu,
                             kernel_initializer=initializer,
                             name='layer')
    layer5 = model5(layer4)

    # Fully connected softmax output layer with 10 nodes
    model6 = tf.layers.Dense(units=10,
                             activation='None',
                             kernel_initializer=initializer,
                             name='layer')
    layer6 = model6(layer5)

    y_pred = tf.nn.softmax(layer6)

    # accuracy
    # from one y_pred one_hot to tag
    y_pred_t = tf.argmax(y_pred, 1)
    # from y one_hot to tag
    y_t = tf.argmax(y, 1)
    # comparison vector between tags (TRUE/FALSE)
    equal = tf.equal(y_pred_t, y_t)
    # average hits
    acc = tf.reduce_mean(tf.cast(equal, tf.float32))

    # loss
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # train_op
    optimizer = tf.train.AdamOptimizer(name='Adam')
    train_op = optimizer.minimize(loss)

    return y_pred, train_op, loss, acc
