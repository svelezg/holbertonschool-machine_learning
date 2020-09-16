#!/usr/bin/env python3
"""contains the ngram_bleu"""


def ngram_bleu(references, sentence, n):
    """
    calculates the n-gram BLEU score for a sentence
    :param references: list of reference translations
    :param sentence: list containing the model proposed sentence
    :param n: size of the n-gram to use for evaluation
    :return: n-gram BLEU score
    """
