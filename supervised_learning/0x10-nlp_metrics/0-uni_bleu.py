#!/usr/bin/env python3
"""
contains the uni_bleu function
based on
https://ariepratama.github.io/Introduction-to-BLEU-in-python/
"""

import numpy as np
import collections


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence
    :param references: list of reference translations
    :param sentence: list containing the model proposed sentence
    :return: the unigram BLEU score

    """
    # sentence dictionary
    sentence_dict = {x: sentence.count(x) for x in sentence}

    # creates the ceiling for later clipping
    references_dict = {}
    for ref in references:
        for gram in ref:
            if gram not in references_dict.keys()\
                    or references_dict[gram] < ref.count(gram):
                references_dict[gram] = ref.count(gram)

    # counts appearances
    appearances = {x: 0 for x in sentence}
    for ref in references:
        for gram in appearances.keys():
            if gram in ref:
                appearances[gram] = sentence_dict[gram]

    # Clipping
    for gram in appearances.keys():
        if gram in references_dict.keys():
            appearances[gram] = min(references_dict[gram], appearances[gram])

    # Precision
    len_trans = len(sentence)
    precision = sum(appearances.values()) / len_trans

    # Brevity penalty
    # closest reference length from translation length
    closest_ref_idx = np.argmin([abs(len(x) - len_trans) for x in references])
    r = len(references[closest_ref_idx])

    if len_trans > r:
        BP = 1
    else:
        BP = np.exp(1 - float(r) / len_trans)

    bleu = BP * precision

    return bleu
