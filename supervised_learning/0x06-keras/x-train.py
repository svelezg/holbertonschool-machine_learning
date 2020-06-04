#!/usr/bin/env python3
"""Contains the train_model function"""

import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import datetime as datetime


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
        trains a model using mini-batch gradient descent
        :param network: model to train
        :param data: numpy.ndarray of shape (m, nx) containing the input data
        :param labels: one-hot numpy.ndarray of shape (m, classes)
            containing the labels of data
        :param batch_size: size of the batch used for mini-batch grad descent
        :param epochs: number of passes through data for mini-batch grad desc
        :param validation_data:  data to validate the model with, if not None
        :param early_stopping: boolean that indicates whether
            early stopping should be used
        :param patience: patience used for early stopping
        :param learning_rate_decay: boolean that indicates whether
            learning rate decay should be used
        :param alpha: initial learning rate
        :param decay_rate: decay rate
        :param save_best: boolean indicating whether to save the model
            after each epoch if it is the best
        :param filepath: file path where the model should be saved
        :param verbose: boolean that determines if output should be printed
            during training
        :param shuffle: boolean that determines whether to shuffle the batches
            every epoch.
            Normally, it is a good idea to shuffle,
            but for reproducibility, we have chosen to set the default to False
        :return: History object generated after training the model
        """
    log_dir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    def lr_schedule(epoch):
        """ updates the learning rate using inverse time decay """
        rate = alpha / (1 + decay_rate * epoch)
        return rate

    callback_list = []

    # tensorboard callback
    tensorboard_callback = K.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callback_list.append(tensorboard_callback)

    # models save callback
    if save_best:
        mcp_save = K.callbacks.ModelCheckpoint(filepath,
                                               save_best_only=True,
                                               monitor='val_loss',
                                               mode='min')
        callback_list.append(mcp_save)

    # learning rate decay callback
    if validation_data and learning_rate_decay:
        lrd = K.callbacks.LearningRateScheduler(lr_schedule,
                                                verbose=1)
        callback_list.append(lrd)

    # early stopping callback
    if validation_data and early_stopping:
        es = K.callbacks.EarlyStopping(monitor='val_loss',
                                       mode='min',
                                       patience=patience)
        callback_list.append(es)

    # training
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle,
                          callbacks=callback_list)

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
