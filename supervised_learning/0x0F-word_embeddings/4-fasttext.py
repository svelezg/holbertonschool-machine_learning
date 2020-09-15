#!/usr/bin/env python3
"""Contains the gensim_to_keras function"""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    creates and trains a genism fastText model
    :param sentences: list of sentences to be trained on
    :param size: dimensionality of the embedding layer
    :param min_count: minimum number of occurrences of a word
        for use in training
    :param negative: size of negative sampling
    :param window: maximum distance between the current and predicted
        word within a sentence
    :param cbow: boolean to determine the training type; True is for CBOW;
        False is for Skip-gram
    :param iterations: number of iterations to train over
    :param seed: seed for the random number generator
    :param workers: number of worker threads to train the model
    :return: model
    """
    # instantiate
    model = FastText(size=size, window=window, min_count=min_count,
                     negative=negative, sg=cbow, seed=seed, workers=workers)

    # vocabulary
    model.build_vocab(sentences=sentences)

    # train
    model.train(sentences=sentences,
                total_examples=len(sentences),
                epochs=iterations)

    return model
