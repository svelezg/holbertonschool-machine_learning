# 0x00\. Q-learning

## Authors
* **Solution:** Santiago Vélez G. [svelez.velezgarcia@gmail.com](svelez.velezgarcia@gmail.com) [@svelezg](https://github.com/svelezg)
* **Problem statement:** Alexa Orrico [Holberton School](https://www.holbertonschool.com/)



## Resources

**Read or watch**:

*   [An introduction to Reinforcement Learning](/rltoken/uSJcrn4-wamVCfbQQtI9EA "An introduction to Reinforcement Learning")
*   [Simple Reinforcement Learning: Q-learning](/rltoken/bmZIQktOMGjnZ6DZj5v00g "Simple Reinforcement Learning: Q-learning")
*   [Markov Decision Processes (MDPs) - Structuring a Reinforcement Learning Problem](/rltoken/km2Nyp6zyAast1k5v9P_wQ "Markov Decision Processes (MDPs) - Structuring a Reinforcement Learning Problem")
*   [Expected Return - What Drives a Reinforcement Learning Agent in an MDP](/rltoken/mM6iGVu8uSr7siZJCM-D-Q "Expected Return - What Drives a Reinforcement Learning Agent in an MDP")
*   [Policies and Value Functions - Good Actions for a Reinforcement Learning Agent](/rltoken/HgOMxHB7SipUwDk6s3ZhUA "Policies and Value Functions - Good Actions for a Reinforcement Learning Agent")
*   [What do Reinforcement Learning Algorithms Learn - Optimal Policies](/rltoken/Pd4kGKXr9Pd0qQ4RO93Xww "What do Reinforcement Learning Algorithms Learn - Optimal Policies")
*   [Q-Learning Explained - A Reinforcement Learning Technique](/rltoken/vj2E0Jizi5qUKn6hLUnVSQ "Q-Learning Explained - A Reinforcement Learning Technique")
*   [Exploration vs. Exploitation - Learning the Optimal Reinforcement Learning Policy](/rltoken/zQNxN36--R7hzP0ktiKOsg "Exploration vs. Exploitation - Learning the Optimal Reinforcement Learning Policy")
*   [OpenAI Gym and Python for Q-learning - Reinforcement Learning Code Project](/rltoken/GMcf0lCJ-SlaF6FSUKaozA "OpenAI Gym and Python for Q-learning - Reinforcement Learning Code Project")
*   [Train Q-learning Agent with Python - Reinforcement Learning Code Project](/rltoken/GE2nKBHgehHdd_XN7lK0Gw "Train Q-learning Agent with Python - Reinforcement Learning Code Project")
*   [Markov Decision Processes](/rltoken/Dz37ih49PpmrJicq_IP3aA "Markov Decision Processes")

**Definitions to skim:**

*   [Reinforcement Learning](/rltoken/z1eKcn91HbmHYtdwYEEXOQ "Reinforcement Learning")
*   [Markov Decision Process](/rltoken/PCdKyrHQRNARmxeSUCiOYQ "Markov Decision Process")
*   [Q-learning](/rltoken/T80msozXZ3wlSmq0ScCvrQ "Q-learning")

**References**:

*   [OpenAI Gym](/rltoken/P8gDRc_PRTeK4okeztvmDQ "OpenAI Gym")
*   [OpenAI Gym: Frozen Lake env](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py "OpenAI Gym: Frozen Lake env")

## Learning Objectives

*   What is a Markov Decision Process?
*   What is an environment?
*   What is an agent?
*   What is a state?
*   What is a policy function?
*   What is a value function? a state-value function? an action-value function?
*   What is a discount factor?
*   What is the Bellman equation?
*   What is epsilon greedy?
*   What is Q-learning?

## Requirements

### General

*   Allowed editors: `vi`, `vim`, `emacs`
*   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
*   Your files will be executed with `numpy` (version 1.15), and `gym` (version 0.7)
*   All your files should end with a new line
*   The first line of all your files should be exactly `#!/usr/bin/env python3`
*   A `README.md` file, at the root of the folder of the project, is mandatory
*   Your code should use the `pycodestyle` style (version 2.4)
*   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
*   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
*   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
*   All your files must be executable
*   **Your code should use the minimum number of operations**

## Installing OpenAI’s Gym

    pip install --user gym



* * *

## Quiz questions

Show







#### Question #0

What is reinforcement learning?

*   <input type="checkbox" data-quiz-question-id="1549" data-quiz-answer-id="1597064459175" disabled="">

    A type of supervised learning, because the rewards supervise the learning

*   <input type="checkbox" data-quiz-question-id="1549" data-quiz-answer-id="1597064470601" disabled="">

    A type of unsupervised learning, because there are no labels for each action

*   <input type="checkbox" data-quiz-question-id="1549" data-quiz-answer-id="1597064530798" disabled="" checked="">

    Its own subcategory of machine learning









#### Question #1

What is an environment?

*   <input type="checkbox" data-quiz-question-id="1550" data-quiz-answer-id="1597064553450" disabled="" checked="">

    The place in which actions can be performed

*   <input type="checkbox" data-quiz-question-id="1550" data-quiz-answer-id="1597064595587" disabled="">

    A description of what the agent sees

*   <input type="checkbox" data-quiz-question-id="1550" data-quiz-answer-id="1597064613427" disabled="">

    A list of actions that can be performed

*   <input type="checkbox" data-quiz-question-id="1550" data-quiz-answer-id="1597064684101" disabled="">

    A description of which actions the agent should perform









#### Question #2

An agent chooses its action based on:

*   <input type="checkbox" data-quiz-question-id="1551" data-quiz-answer-id="1597064723091" disabled="" checked="">

    The current state

*   <input type="checkbox" data-quiz-question-id="1551" data-quiz-answer-id="1597064777168" disabled="" checked="">

    The value function

*   <input type="checkbox" data-quiz-question-id="1551" data-quiz-answer-id="1597064804098" disabled="" checked="">

    The policy function

*   <input type="checkbox" data-quiz-question-id="1551" data-quiz-answer-id="1597064811945" disabled="" checked="">

    The previous reward









#### Question #3

What is a policy function?

*   <input type="checkbox" data-quiz-question-id="1553" data-quiz-answer-id="1597064872304" disabled="">

    A description of how the agent should be rewarded

*   <input type="checkbox" data-quiz-question-id="1553" data-quiz-answer-id="1597064897527" disabled="" checked="">

    A description of how the agent should behave

*   <input type="checkbox" data-quiz-question-id="1553" data-quiz-answer-id="1597064922926" disabled="">

    A description of how the agent could be rewarded in the future

*   <input type="checkbox" data-quiz-question-id="1553" data-quiz-answer-id="1597064941601" disabled="" checked="">

    A function that is learned

*   <input type="checkbox" data-quiz-question-id="1553" data-quiz-answer-id="1597064983883" disabled="">

    A function that is set at the beginning









#### Question #4

What is a value function?

*   <input type="checkbox" data-quiz-question-id="1554" data-quiz-answer-id="1597064976871" disabled="">

    A description of how the agent should be rewarded

*   <input type="checkbox" data-quiz-question-id="1554" data-quiz-answer-id="1597065040069" disabled="">

    A description of how the agent should behave

*   <input type="checkbox" data-quiz-question-id="1554" data-quiz-answer-id="1597065041922" disabled="" checked="">

    A description of how the agent could be rewarded in the future

*   <input type="checkbox" data-quiz-question-id="1554" data-quiz-answer-id="1597065043314" disabled="" checked="">

    A function that is learned

*   <input type="checkbox" data-quiz-question-id="1554" data-quiz-answer-id="1597065044824" disabled="">

    A function that is set at the beginning









#### Question #5

What is epsilon-greedy?

*   <input type="checkbox" data-quiz-question-id="1555" data-quiz-answer-id="1597065092154" disabled="">

    A type of policy function

*   <input type="checkbox" data-quiz-question-id="1555" data-quiz-answer-id="1597065108393" disabled="">

    A type of value function

*   <input type="checkbox" data-quiz-question-id="1555" data-quiz-answer-id="1597065116708" disabled="">

    A way to balance policy and value functions

*   <input type="checkbox" data-quiz-question-id="1555" data-quiz-answer-id="1597065133195" disabled="" checked="">

    A balance exploration and exploitation









#### Question #6

What is Q-learning?

*   <input type="checkbox" data-quiz-question-id="1556" data-quiz-answer-id="1597065211073" disabled="" checked="">

    A reinforcement learning algorithm

*   <input type="checkbox" data-quiz-question-id="1556" data-quiz-answer-id="1597065222265" disabled="">

    A deep reinforcement learning algorithm

*   <input type="checkbox" data-quiz-question-id="1556" data-quiz-answer-id="1597065232361" disabled="" checked="">

    A value-based learning algorithm

*   <input type="checkbox" data-quiz-question-id="1556" data-quiz-answer-id="1597065318628" disabled="">

    A policy-based learning algorithm

*   <input type="checkbox" data-quiz-question-id="1556" data-quiz-answer-id="1597065330302" disabled="">

    A model-based approach




* * *

## Tasks




#### 0\. Load the Environment 

Write a function `def load_frozen_lake(desc=None, map_name=None, is_slippery=False):` that loads the pre-made `FrozenLakeEnv` evnironment from OpenAI’s `gym`:

*   `desc` is either `None` or a list of lists containing a custom description of the map to load for the environment
*   `map_name` is either `None` or a string containing the pre-made map to load
*   _Note: If both `desc` and `map_name` are `None`, the environment will load a randomly generated 8x8 map_
*   `is_slippery` is a boolean to determine if the ice is slippery
*   Returns: the environment

```
    
    $ ./0-main.py
    [[b'S' b'F' b'F' b'F' b'F' b'F' b'F' b'H']
     [b'H' b'F' b'F' b'F' b'F' b'H' b'F' b'F']
     [b'F' b'H' b'F' b'H' b'H' b'F' b'F' b'F']
     [b'F' b'F' b'F' b'H' b'F' b'F' b'F' b'F']
     [b'F' b'F' b'F' b'F' b'F' b'F' b'H' b'F']
     [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
     [b'F' b'F' b'F' b'F' b'H' b'F' b'F' b'F']
     [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'G']]
    [(1.0, 0, 0.0, False)]
    [[b'S' b'F' b'H' b'F' b'H' b'F' b'H' b'F']
     [b'H' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
     [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
     [b'F' b'H' b'F' b'F' b'F' b'F' b'F' b'F']
     [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'H']
     [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'H']
     [b'F' b'F' b'H' b'F' b'H' b'F' b'H' b'F']
     [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'G']]
    [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 8, 0.0, True)]
    [[b'S' b'F' b'F']
     [b'F' b'H' b'H']
     [b'F' b'F' b'G']]
    [[b'S' b'F' b'F' b'F']
     [b'F' b'H' b'F' b'H']
     [b'F' b'F' b'F' b'H']
     [b'H' b'F' b'F' b'G']]
    $
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `reinforcement_learning/0x00-q_learning`
*   File: `0-load_env.py`





#### 1\. Initialize Q-table 

Write a function `def q_init(env):` that initializes the Q-table:

*   `env` is the `FrozenLakeEnv` instance
*   Returns: the Q-table as a `numpy.ndarray` of zeros

```
    
    $ ./1-main.py
    (64, 4)
    (64, 4)
    (9, 4)
    (16, 4)
    $
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `reinforcement_learning/0x00-q_learning`
*   File: `1-q_init.py````





#### 2\. Epsilon Greedy 

Write a function `def epsilon_greedy(Q, state, epsilon):` that uses epsilon-greedy to determine the next action:

*   `Q` is a `numpy.ndarray` containing the q-table
*   `state` is the current state
*   `epsilon` is the epsilon to use for the calculation
*   You should sample `p` with `numpy.random.uniformn` to determine if your algorithm should explore or exploit
*   If exploring, you should pick the next action with `numpy.random.randint` from all possible actions
*   Returns: the next action index

```
    
    $ ./2-main.py
    2
    0
    $
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `reinforcement_learning/0x00-q_learning`
*   File: `2-epsilon_greedy.py````















#### 3\. Q-learning 

Write the function `def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):` that performs Q-learning:

*   `env` is the `FrozenLakeEnv` instance
*   `Q` is a `numpy.ndarray` containing the Q-table
*   `episodes` is the total number of episodes to train over
*   `max_steps` is the maximum number of steps per episode
*   `alpha` is the learning rate
*   `gamma` is the discount rate
*   `epsilon` is the initial threshold for epsilon greedy
*   `min_epsilon` is the minimum value that `epsilon` should decay to
*   `epsilon_decay` is the decay rate for updating `epsilon` between episodes
*   When the agent falls in a hole, the reward should be updated to be `-1`
*   Returns: `Q, total_rewards`
    *   `Q` is the updated Q-table
    *   `total_rewards` is a list containing the rewards per episode

```
    
    $ ./3-main.py
    [[ 0.96059593  0.970299    0.95098488  0.96059396]
     [ 0.96059557 -0.77123208  0.0094072   0.37627228]
     [ 0.18061285 -0.1         0\.          0\.        ]
     [ 0.97029877  0.9801     -0.99999988  0.96059583]
     [ 0\.          0\.          0\.          0\.        ]
     [ 0\.          0\.          0\.          0\.        ]
     [ 0.98009763  0.98009933  0.99        0.9702983 ]
     [ 0.98009922  0.98999782  1\.         -0.99999952]
     [ 0\.          0\.          0\.          0\.        ]]
    500 : 0.812
    1000 : 0.88
    1500 : 0.9
    2000 : 0.9
    2500 : 0.88
    3000 : 0.844
    3500 : 0.892
    4000 : 0.896
    4500 : 0.852
    5000 : 0.928
    $
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `reinforcement_learning/0x00-q_learning`
*   File: `3-q_learning.py````







#### 4\. Play 

Write a function `def play(env, Q, max_steps=100):` that has the trained agent play an episode:

*   `env` is the `FrozenLakeEnv` instance
*   `Q` is a `numpy.ndarray` containing the Q-table
*   `max_steps` is the maximum number of steps in the episode
*   Each state of the board should be displayed via the console
*   You should always exploit the Q-table
*   Returns: the total rewards for the episode

```
    
    $ ./4-main.py

    `S`FF
    FHH
    FFG
      (Down)
    SFF
    `F`HH
    FFG
      (Down)
    SFF
    FHH
    `F`FG
      (Right)
    SFF
    FHH
    F`F`G
      (Right)
    SFF
    FHH
    FF`G`
    1.0
    $
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `reinforcement_learning/0x00-q_learning`
*   File: `4-play.py`
