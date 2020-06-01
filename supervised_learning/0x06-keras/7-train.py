#!/usr/bin/env python3
"""Contains the train_model function"""

import tensorflow as tf
import matplotlib.pyplot as plt


def train_model(network, data, labels,
                batch_size, epochs,
                validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
        trains a model using mini-batch gradient descent
        :param network: model to train
        :param data: numpy.ndarray of shape (m, nx) containing the input data
        :param labels: one-hot numpy.ndarray of shape (m, classes)
            containing the labels of data
        :param batch_size: size of the batch used for mini-batch grad desc
        :param epochs: number of passes through data for mini-batch grad desc
        :param validation_data:  data to validate the model with, if not None
        :param early_stopping: boolean that indicates whether
            early stopping should be used
        :param patience: patience used for early stopping
        :param learning_rate_decay: boolean that indicates whether
            learning rate decay should be used
        :param alpha: initial learning rate
        :param decay_rate: decay rate
        :param verbose: boolean that determines if output should be
            printed during training
        :param shuffle: boolean that determines whether to shuffle
            the batches every epoch.
        :return: History object generated after training the model
        """
    def learning_rate(epoch):
        """ updates the learning rate using inverse time decay """
        return alpha / (1 + decay_rate * epoch)
        # return alpha * tf.math.exp(decay_rate * (1 - epoch))

    cb_list = []

    if validation_data and learning_rate_decay:
        lrd = tf.keras.callbacks.LearningRateScheduler(learning_rate,
                                                       verbose=1)
        cb_list.append(lrd)

    if validation_data and early_stopping:
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min',
                                              patience=patience)
        cb_list.append(es)
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle,
                          callbacks=cb_list
                          )

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return history
