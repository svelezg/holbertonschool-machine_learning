#!/usr/bin/env python3
"""Contains the word2vec_model function"""

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import tensorflow.keras as keras


def word2vec_model(sentences, size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """
    creates and trains a gensim word2vec model
    :param sentences: list of sentences to be trained on
    :param size: the dimensionality of the embedding layer
    :param min_count: the minimum number of occurrences of a word
        for use in training
    :param window: maximum distance between the current and predicted word
        within a sentence
    :param negative: size of negative sampling
    :param cbow: boolean to determine the training type; True is for CBOW;
        False is for Skip-gram
    :param iterations: number of iterations to train over
    :param seed: seed for the random number generator
    :param workers: number of worker threads to train the model
    :return: trained model
    """
    model = Word2Vec(sentences, size=size,
                     window=window, min_count=min_count,
                     negative=negative, workers=workers,
                     sg=cbow, seed=seed, iter=iterations)

    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
