#!/usr/bin/env python3
"""Contains the gensim_to_keras function"""


def gensim_to_keras(model):
    """
    converts a gensim word2vec model to a keras Embedding layer
    :param model: trained gensim word2vec models
    :return: trainable keras Embedding
    """
