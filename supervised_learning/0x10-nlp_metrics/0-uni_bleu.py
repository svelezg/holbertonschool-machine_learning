#!/usr/bin/env python3
"""contains the uni_bleu function"""


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence
    :param references: list of reference translations
    :param sentence: list containing the model proposed sentence
    :return: the unigram BLEU score
    """
