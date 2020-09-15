#!/usr/bin/env python3
"""Contains the gensim_to_keras function"""

from gensim.models import Word2Vec
import tensorflow.keras as keras


def gensim_to_keras(model):
    """
    converts a gensim word2vec model to a keras Embedding layer
    :param model: trained gensim word2vec models
    :return: trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
