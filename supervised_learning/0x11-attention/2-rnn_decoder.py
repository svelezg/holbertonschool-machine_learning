#!/usr/bin/env python3
"""contains the RNNEncoder class"""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """decode for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        :param vocab: integer representing the size of the output vocabulary
        :param embedding: integer representing the dimensionality
            of the embedding vector
        :param units: integer representing the number
            of hidden units in the RNN cell
        :param batch: integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)

        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        :param x: tensor of shape (batch, 1)
            containing the previous word in the target sequence
            as an index of the target vocabulary
        :param s_prev: tensor of shape (batch, units)
            containing the previous decoder hidden state
        :param hidden_states: tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder
        :return: y, s
            y is a tensor of shape (batch, vocab)
                containing the output word as a one hot vector
                in the target vocabulary
            s is a tensor of shape (batch, units)
                containing the new decoder hidden state
        """
        batch, units = s_prev.shape

        # instantiate SelfAttention
        attention = SelfAttention(units)

        # calculate context and weights
        context, weights = attention(s_prev, hidden_states)

        # expand context dims to match embeddings
        # (input_length)
        exp_context = tf.expand_dims(context, 1)

        # calculate embeddings
        embeddings = self.embedding(x)

        concat_input = tf.concat([exp_context,
                                  embeddings],
                                 axis=-1)

        # GRU takes a 3D input
        outputs, hidden = self.gru(concat_input)

        # reshape output (suppress input_seq_len)
        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))

        # final layer output word as a one hot vector
        # in the target vocabulary
        y = self.F(outputs)

        return y, hidden
