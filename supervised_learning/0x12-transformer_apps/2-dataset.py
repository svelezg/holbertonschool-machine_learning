#!/usr/bin/env python3
"""contains the Dataset class"""

import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """loads and preps a dataset for machine translation"""

    def __init__(self):
        """Class constructor"""
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train, self.data_valid = examples['train'], \
            examples['validation']

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset
        :param data: data is a tf.data.Dataset whose examples are formatted as
            a tuple (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence
        :return: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in data),
                target_vocab_size=2 ** 15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in data),
                target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        encodes a translation into tokens
        :param pt: tf.Tensor containing the Portuguese sentence
        :param en: tf.Tensor containing the corresponding English sentence
        :return: pt_tokens, en_tokens
            pt_tokens is a tf.Tensor containing the Portuguese tokens
            en_tokens is a tf.Tensor containing the English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        acts as a tensorflow wrapper for the encode instance method
        :param pt: tf.Tensor containing the Portuguese sentence
        :param en: tf.Tensor containing the corresponding English sentence
        :return:
        """
        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
