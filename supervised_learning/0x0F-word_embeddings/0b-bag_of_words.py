#!/usr/bin/env python3
"""Contains the bag_of_words function"""

import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix
    :param sentences: list of sentences to analyze
    :param vocab: list of the vocabulary words to use for the analysis
    :return: embeddings, features
        embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features is a list of the features used for embeddings
    """
    s = len(sentences)

    if vocab is None:
        vocab = []
        for sentence in sentences:
            words = sentence.split()
            for word in words:
                clean = ''.join([c for c in word.lower() if c.isalpha()])
                if clean not in vocab:
                    vocab.append(clean)
        vocab.sort()

    f = len(vocab)

    embeddings = np.zeros((s, f), np.int8)

    for i in range(s):
        sentence = sentences[i]
        for word in sentence.split():
            clean = ''.join([c for c in word.lower() if c.isalpha()])
            index = vocab.index(clean)
            embeddings[i, index] += int(1)

    return embeddings, vocab
