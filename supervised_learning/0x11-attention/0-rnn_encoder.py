#!/usr/bin/env python3
"""contains the RNNEncoder class"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """encoder for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        :param vocab: integer representing the size of the input vocabulary
        :param embedding: integer representing the dimensionality
            of the embedding vector
        :param units: integer representing the number
            of hidden units in the RNN cell
        :param batch: integer representing the batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros
        :return: tensor of shape (batch, units)
        containing the initialized hidden states
        """
        initializer = tf.keras.initializers.Zeros()
        return initializer(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
        :param x: tensor of shape (batch, input_seq_len)
            containing the input to the encoder layer
            as word indices within the vocabulary
        :param initial: tensor of shape (batch, units)
            containing the initial hidden state
        :return: outputs, hidden
            outputs is a tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder
            hidden is a tensor of shape (batch, units)
                containing the last hidden state of the encoder
        """
        embeddings = self.embedding(x)
        outputs, hidden = self.gru(embeddings,
                                   initial_state=initial)

        return outputs, hidden
