#!/usr/bin/env python3
"""contains the q_init function"""

import numpy as np


def q_init(env):
    """
    initializes the Q-table
    :param env: the FrozenLakeEnv instance
    :return: Q-table as a numpy.ndarray of zeros
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    return Q
