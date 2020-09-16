# 0x10\. Natural Language Processing - Evaluation Metrics


## Authors
* **Solution:** Santiago Vélez G. [svelez.velezgarcia@gmail.com](svelez.velezgarcia@gmail.com) [@svelezg](https://github.com/svelezg)
* **Problem statement:** Alexa Orrico [Holberton School](https://www.holbertonschool.com/)




## Resources

**Read or watch:**

*   [7 Applications of Deep Learning for Natural Language Processing](/rltoken/EFeppnrszrEGza6nrymxgQ "7 Applications of Deep Learning for Natural Language Processing")
*   [10 Applications of Artificial Neural Networks in Natural Language Processing](/rltoken/Pcs54fB9zpZZWlMH_OamfQ "10 Applications of Artificial Neural Networks in Natural Language Processing")
*   [A Gentle Introduction to Calculating the BLEU Score for Text in Python](/rltoken/lC85P6iX492bGuBncUNwiw "A Gentle Introduction to Calculating the BLEU Score for Text in Python")
*   [Bleu Score](/rltoken/lT-MBM6w7AjXPIiZoPKR_A "Bleu Score")
*   [Evaluating Text Output in NLP: BLEU at your own risk](/rltoken/T1wHe4YsSy5GtsEJNQB4bg "Evaluating Text Output in NLP: BLEU at your own risk")
*   [What Is ROUGE And How It Works For Evaluation Of Summarization Tasks?](/rltoken/iaNCKPSMskZjUcw1hv8-pw "What Is ROUGE And How It Works For Evaluation Of Summarization Tasks?")
*   [Evaluating Summaries ROUGE](/rltoken/8InyFuc-569qD7XbKchiHg "Evaluating Summaries ROUGE")
*   [Evaluation and Perplexity](/rltoken/Mgyoxa8c6WvpFJaHFxqlQQ "Evaluation and Perplexity")
*   [Predicting the Next Word: Back-Off Language Modeling](/rltoken/rQ6MC0ZkpPjHBCEE3hpl4A "Predicting the Next Word: Back-Off Language Modeling")

**Definitions to skim**

*   [BLEU](/rltoken/njmmpbMuP0cPnnWwbFpj3A "BLEU")
*   [ROUGE](/rltoken/BJK2tEo1kVYXytMDoVF9fQ "ROUGE")
*   [Perplexity](/rltoken/MayHONfLeczBB8qWvaDrkQ "Perplexity")

**References:**

*   [BLEU: a Method for Automatic Evaluation of Machine Translation (2002)](/rltoken/F8q42L0L3uzw6RLlLj1QsQ "BLEU: a Method for Automatic Evaluation of Machine Translation (2002)")
*   [ROUGE: A Package for Automatic Evaluation of Summaries (2004)](/rltoken/gDkBzTZvzANW2xRpV5Ea5g "ROUGE: A Package for Automatic Evaluation of Summaries (2004)")

## Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](/rltoken/1x1paga16dECc8bSS9ddYQ "explain to anyone"), **without the help of Google**:

### General

*   What are the applications of natural language processing?
*   What is a BLEU score?
*   What is a ROUGE score?
*   What is perplexity?
*   When should you use one evaluation metric over another?

## Requirements

### General

*   Allowed editors: `vi`, `vim`, `emacs`
*   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
*   Your files will be executed with `numpy` (version 1.15)
*   All your files should end with a new line
*   The first line of all your files should be exactly `#!/usr/bin/env python3`
*   All of your files must be executable
*   A `README.md` file, at the root of the folder of the project, is mandatory
*   Your code should follow the `pycodestyle` style (version 2.4)
*   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
*   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
*   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
*   You are not allowed to use the `nltk` module



* * *

## Quiz questions

Show







#### Question #0

The BLEU score measures:

*   <input type="checkbox" data-quiz-question-id="1171" data-quiz-answer-id="1594901354088" disabled="">

    A model’s accuracy

*   <input type="checkbox" data-quiz-question-id="1171" data-quiz-answer-id="1594901355918" disabled="" checked="">

    A model’s precision

*   <input type="checkbox" data-quiz-question-id="1171" data-quiz-answer-id="1594901357151" disabled="">

    A model’s recall

*   <input type="checkbox" data-quiz-question-id="1171" data-quiz-answer-id="1594901358677" disabled="">

    A model’s perplexity









#### Question #1

The ROUGE score measures:

*   <input type="checkbox" data-quiz-question-id="1172" data-quiz-answer-id="1594901403019" disabled="">

    A model’s accuracy

*   <input type="checkbox" data-quiz-question-id="1172" data-quiz-answer-id="1594901414214" disabled="" checked="">

    A model’s precision

*   <input type="checkbox" data-quiz-question-id="1172" data-quiz-answer-id="1594901421842" disabled="" checked="">

    A model’s recall

*   <input type="checkbox" data-quiz-question-id="1172" data-quiz-answer-id="1594901451064" disabled="">

    A model’s perplexity









#### Question #2

Perplexity measures:

*   <input type="checkbox" data-quiz-question-id="1173" data-quiz-answer-id="1594901553611" disabled="">

    The accuracy of a prediction

*   <input type="checkbox" data-quiz-question-id="1173" data-quiz-answer-id="1594901572049" disabled="" checked="">

    The branching factor of a prediction

*   <input type="checkbox" data-quiz-question-id="1173" data-quiz-answer-id="1594901683377" disabled="">

    A prediction’s recall

*   <input type="checkbox" data-quiz-question-id="1173" data-quiz-answer-id="1594901689404" disabled="">

    A prediction’s accuracy









#### Question #3

The BLEU score was designed for:

*   <input type="checkbox" data-quiz-question-id="1174" data-quiz-answer-id="1594901820841" disabled="">

    Sentiment Analysis

*   <input type="checkbox" data-quiz-question-id="1174" data-quiz-answer-id="1594901830774" disabled="" checked="">

    Machine Translation

*   <input type="checkbox" data-quiz-question-id="1174" data-quiz-answer-id="1594901837617" disabled="">

    Question-Answering

*   <input type="checkbox" data-quiz-question-id="1174" data-quiz-answer-id="1594901861763" disabled="">

    Document Summarization









#### Question #4

What are the shortcomings of the BLEU score?

*   <input type="checkbox" data-quiz-question-id="1175" data-quiz-answer-id="1594901878460" disabled="" checked="">

    It cannot judge grammatical accuracy

*   <input type="checkbox" data-quiz-question-id="1175" data-quiz-answer-id="1594901879890" disabled="" checked="">

    It cannot judge meaning

*   <input type="checkbox" data-quiz-question-id="1175" data-quiz-answer-id="1594901881115" disabled="" checked="">

    It does not work with languages that lack word boundaries

*   <input type="checkbox" data-quiz-question-id="1175" data-quiz-answer-id="1594901882506" disabled="" checked="">

    A higher score is not necessarily indicative of a better translation







* * *

## Tasks





#### 0\. Unigram BLEU score 

Write the function `def uni_bleu(references, sentence):` that calculates the unigram BLEU score for a sentence:

*   `references` is a list of reference translations
    *   each reference translation is a list of the words in the translation
*   `sentence` is a list containing the model proposed sentence
*   Returns: the unigram BLEU score

    
    $ ./0-main.py
    0.6549846024623855
    $

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `supervised_learning/0x10-nlp_metrics`
*   File: `0-uni_bleu.py`





















#### 1\. N-gram BLEU score 

Write the function `def ngram_bleu(references, sentence, n):` that calculates the n-gram BLEU score for a sentence:

*   `references` is a list of reference translations
    *   each reference translation is a list of the words in the translation
*   `sentence` is a list containing the model proposed sentence
*   `n` is the size of the n-gram to use for evaluation
*   Returns: the n-gram BLEU score

    
    $ ./1-main.py
    0.6140480648084865
    $

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `supervised_learning/0x10-nlp_metrics`
*   File: `1-ngram_bleu.py`









#### 2\. Cumulative N-gram BLEU score 

Write the function `def cumulative_bleu(references, sentence, n):` that calculates the cumulative n-gram BLEU score for a sentence:

*   `references` is a list of reference translations
    *   each reference translation is a list of the words in the translation
*   `sentence` is a list containing the model proposed sentence
*   `n` is the size of the largest n-gram to use for evaluation
*   All n-gram scores should be weighted evenly
*   Returns: the cumulative n-gram BLEU score

    
    $ ./2-main.py
    0.5475182535069453
    $

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `supervised_learning/0x10-nlp_metrics`
*   File: `2-cumulative_bleu.py`


