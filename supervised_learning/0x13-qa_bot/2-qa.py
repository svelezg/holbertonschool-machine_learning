#!/usr/bin/env python3
"""
contains the function answer_loop
"""

from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf


def question_answer(question, reference):
    """
    :param question: string containing the question to answer
    :param reference: string containing the reference document
    from which to find the answer
    :return: string containing the answer
    """

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = TFAutoModelForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad",
        return_dict=True)

    inputs = tokenizer(question, reference,
                       add_special_tokens=True, return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    output = model(inputs)

    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(
        output.start_logits, axis=1
    ).numpy()[0]

    # Get the most likely end of answer with the argmax of the score
    answer_end = (
            tf.argmax(output.end_logits, axis=1) + 1
    ).numpy()[0]

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer


def answer_loop(reference):
    """
    answers questions from a reference text
    :param reference: reference text
    :return:
    """

    while 1:
        question = input("Q: ")

        if question.lower() in ['exit', 'goodbye', 'bye']:
            print("A: Goodbye")
            exit(0)
        else:
            answer = question_answer(question, reference)

            if answer not in ['[CLS]']:
                print("A:", answer)
            else:
                print("A:", 'Sorry, I do not understand your question.')
