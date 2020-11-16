# 0x13\. QA Bot

## Authors
* **Solution:** Santiago Vélez G. [svelez.velezgarcia@gmail.com](svelez.velezgarcia@gmail.com) [@svelezg](https://github.com/svelezg)
* **Problem statement:** Alexa Orrico 



## Resources

**Read or watch**:

*   [Improving Language Understanding by Generative Pre-Training (2018)](/rltoken/isif62esIXAxfKrVWRFUvA "Improving Language Understanding by Generative Pre-Training (2018)")
*   [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](/rltoken/tFF7DtiDKAHOB_AnliLQ-w "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)")
*   [SQuAD 2.0](/rltoken/TWw2zYrbEv1A6Xp7Q5xeIg "SQuAD 2.0")
*   [Know What You Don’t Know: Unanswerable Questions for SQuAD (2018)](/rltoken/JDS8l-V-AyRMgZ-VWCR4mg "Know What You Don’t Know: Unanswerable Questions for SQuAD (2018)")
*   [GLUE Benchmark](/rltoken/G5RYcXocPNTEGSeiWFfT9Q "GLUE Benchmark")
*   [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding (2019)](/rltoken/58OooeuXuihK66vnEQL5og "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding (2019)")
*   [Speech-transformer: A no-recurrence sequence-to-sequence model for speech recognition (2018)](/rltoken/t0l-uUrcWnFizy9y6ySnOg "Speech-transformer: A no-recurrence sequence-to-sequence model for speech recognition (2018)")

**More recent papers in NLP:**

*   [Generating Long Sequences with Sparse Transformers (2019)](/rltoken/fLho-pVtLacRpYcoeKQlbw "Generating Long Sequences with Sparse Transformers (2019)")
*   [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (2019)](/rltoken/DNjHiFbB_cOVhIB7usnx8A "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (2019)")
*   [XLNet: Generalized Autoregressive Pretraining for Language Understanding (2019)](/rltoken/fr6iKOeQ2J2I3m-2mnVk5w "XLNet: Generalized Autoregressive Pretraining for Language Understanding (2019)")
*   [Language Models are Unsupervised Multitask Learners (GPT-2, 2019)](/rltoken/4eTrqxWdepAKneRJsjaD8g "Language Models are Unsupervised Multitask Learners (GPT-2, 2019)")
*   [Language Models are Few-Shot Learners (GPT-3, 2020)](/rltoken/-LbOp0yRvuSn1Hqzd2OF2Q "Language Models are Few-Shot Learners (GPT-3, 2020)")
*   [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations (2020)](/rltoken/s4A59Pf2b2QjWT8NjgWB8g "ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations (2020)")

To keep up with the newest papers and their code bases go to [paperswithcode.com](/rltoken/Glv09z1yOkMl0Gsl0zBHjg "paperswithcode.com"). For example, check out the [raked list of state of the art models for Language Modelling on Penn Treebank](/rltoken/DZKmWejiXC0czbR9j32N_g "raked list of state of the art models for Language Modelling on Penn Treebank").

## Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](/rltoken/gQzVqyxWWUByblpobq6iTA "explain to anyone"), **without the help of Google**:

### General

*   What is Question-Answering?
*   What is Semantic Search?
*   What is BERT?
*   How to develop a QA chatbot
*   How to use the `transformers` library
*   How to use the `tensorflow-hub` library

## Requirements

### General

*   Allowed editors: `vi`, `vim`, `emacs`
*   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.6.12)
*   Your files will be executed with `numpy` (version 1.18) and `tensorflow` (version 2.3)
*   All your files should end with a new line
*   The first line of all your files should be exactly `#!/usr/bin/env python3`
*   All of your files must be executable
*   A `README.md` file, at the root of the folder of the project, is mandatory
*   Your code should follow the `pycodestyle` style (version 2.4)
*   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
*   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
*   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)

## Upgrade to Tensorflow 2.3

`pip install --user tensorflow==2.3`

## Install Tensorflow Hub

`pip install --user tensorflow-hub`

## Install Transformers

`pip install --user transformers`

## Zendesk Articles

For this project, we will be using a collection of Holberton USA Zendesk Articles, [ZendeskArticles.zip](/rltoken/ujQQ_o24BufzUPyWrR7IGA "ZendeskArticles.zip").



* * *

## Tasks



#### 0\. Question Answering 

Write a function `def question_answer(question, reference):` that finds a snippet of text within a reference document to answer a question:

*   `question` is a string containing the question to answer
*   `reference` is a string containing the reference document from which to find the answer
*   Returns: a string containing the answer
*   If no answer is found, return `None`
*   Your function should use the `bert-uncased-tf2-qa` model from the `tensorflow-hub` library
*   Your function should use the pre-trained `BertTokenizer`, `bert-large-uncased-whole-word-masking-finetuned-squad`, from the `transformers` library

```
    
    $ ./0-main.py
    from 9 : 00 am to 3 : 00 pm
    $
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `supervised_learning/0x13-qa_bot`
*   File: `0-qa.py`












#### 1\. Create the loop 

Create a script that takes in input from the user with the prompt `Q:` and prints `A:` as a response. If the user inputs `exit`, `quit`, `goodbye`, or `bye`, case insensitive, print `A: Goodbye` and exit.

    $ ./1-loop.py
    Q: Hello
    A:
    Q: How are you?
    A:
    Q: BYE
    A: Goodbye
    $

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x13-qa_bot`
*   File: `1-loop.py`












#### 2\. Answer Questions 

Based on the previous tasks, write a function `def answer_loop(reference):` that answers questions from a reference text:

*   `reference` is the reference text
*   If the answer cannot be found in the reference text, respond with `Sorry, I do not understand your question.`

```
    
    $ ./2-main.py
    Q: When are PLDs?
    A: from 9 : 00 am to 3 : 00 pm
    Q: What are Mock Interviews?
    A: Sorry, I do not understand your question.
    Q: What does PLD stand for?
    A: peer learning days
    Q: EXIT
    A: Goodbye
    $
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `supervised_learning/0x13-qa_bot`
*   File: `2-qa.py`












#### 3\. Semantic Search 

Write a function `def semantic_search(corpus_path, sentence):` that performs semantic search on a corpus of documents:

*   `corpus_path` is the path to the corpus of reference documents on which to perform semantic search
*   `sentence` is the sentence from which to perform semantic search
*   Returns: the reference text of the document most similar to `sentence`

```
    
    $ ./ 3-main.py
    PLD Overview
    Peer Learning Days (PLDs) are a time for you and your peers to ensure that each of you understands the concepts you've encountered in your projects, as well as a time for everyone to collectively grow in technical, professional, and soft skills. During PLD, you will collaboratively review prior projects with a group of cohort peers.
    PLD Basics
    PLDs are mandatory on-site days from 9:00 AM to 3:00 PM. If you cannot be present or on time, you must use a PTO. 
    No laptops, tablets, or screens are allowed until all tasks have been whiteboarded and understood by the entirety of your group. This time is for whiteboarding, dialogue, and active peer collaboration. After this, you may return to computers with each other to pair or group program. 
    Peer Learning Days are not about sharing solutions. This doesn't empower peers with the ability to solve problems themselves! Peer learning is when you share your thought process, whether through conversation, whiteboarding, debugging, or live coding. 
    When a peer has a question, rather than offering the solution, ask the following:
    "How did you come to that conclusion?"
    "What have you tried?"
    "Did the man page give you a lead?"
    "Did you think about this concept?"
    Modeling this form of thinking for one another is invaluable and will strengthen your entire cohort.
    Your ability to articulate your knowledge is a crucial skill and will be required to succeed during technical interviews and through your career. 
    $
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `supervised_learning/0x13-qa_bot`
*   File: `3-semantic_search.py`





#### 4\. Multi-reference Question Answering 

Based on the previous tasks, write a function `def question_answer(coprus_path):` that answers questions from multiple reference texts:

*   `corpus_path` is the path to the corpus of reference documents

```
    
    $ ./4-main.py
    Q: When are PLDs?
    A: from 9 : 00 am to 3 : 00 pm
    Q: What are Mock Interviews?
    A: help you train for technical interviews
    Q: What does PLD stand for?
    A: peer learning days
    Q: goodbye
    A: Goodbye
    $
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `supervised_learning/0x13-qa_bot`
*   File: `4-qa.py`
