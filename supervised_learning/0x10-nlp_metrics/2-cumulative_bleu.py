#!/usr/bin/env python3
""" contains the def cumulative_bleu(references, sentence, n):"""


def cumulative_bleu(references, sentence, n):
    """
    calculates the cumulative n-gram BLEU score for a sentence
    :param references: list of reference translations
    :param sentence: list containing the model proposed sentence
    :param n: size of the largest n-gram to use for evaluation
    :return: cumulative n-gram BLEU score
    """
