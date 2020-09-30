#!/usr/bin/env python3
"""contains the train_transformer function"""

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    creates and trains a transformer model for machine translation of
        Portuguese to English using our previously created dataset
    :param N: number of blocks in the encoder and decoder
    :param dm: dimensionality of the model
    :param h:  number of heads
    :param hidden: number of hidden units in the fully connected layers
    :param max_len: maximum number of tokens per sequence
    :param batch_size: batch size for training
    :param epochs: number of epochs to train for
    :return:
    """
