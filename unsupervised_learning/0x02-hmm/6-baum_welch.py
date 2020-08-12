#!/usr/bin/env python3
"""contains the baum_welch function"""

import numpy as np

forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    performs the Baum-Welch algorithm for a hidden markov model
    :param Observations: numpy.ndarray of shape (T,)
        that contains the index of the observation
        T is the number of observations
    :param Transition: numpy.ndarray of shape (M, M)
        that contains the initialized transition probabilities
        M is the number of hidden states
    :param Emission: numpy.ndarray of shape (M, N)
        that contains the initialized emission probabilities
        N is the number of output states
    :param Initial: numpy.ndarray of shape (M, 1)
        that contains the initialized starting probabilities
    :param iterations: number of times expectation-maximization
        should be performed
    :return: the converged Transition, Emission, or None, None on failure
    """
    # type and len(dim) conditions
    if not isinstance(Observations, np.ndarray)\
            or len(Observations.shape) != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    # dim conditions
    T = Observations.shape[0]

    N, M = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    # stochastic
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    print('0', '\n', Transition, '\n', Emission)

    for n in range(iterations):
        P, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)
        xi = np.zeros((N, N, T - 1))

        for t in range(T - 1):
            # denominator = alpha[:, t]
            denominator = P
            for i in range(N):
                f = alpha[i, t]
                g = Transition[i, :]
                h = Emission[:, Observations[t + 1]]
                j = beta[i, t + 1]

                numerator =\
                    alpha[i, t] *\
                    Transition[i, :] *\
                    Emission[:, Observations[t + 1]] *\
                    beta[i, t + 1]
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        # gamma1 = alpha * beta / denominator.reshape(denominator.shape[0], 1)
        # gamma = gamma1[:, :T-1]

        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # add additional T'th element in gamma
        xi_sum = np.sum(xi[:, :, T - 2], axis=0)
        xi_sum = xi_sum.reshape((xi_sum.shape[0], 1))
        gamma = np.hstack((gamma, xi_sum))

        denominator = np.sum(gamma, axis=1)

        for i in range(M):
            index = np.where(Observations.reshape(T, 1) == i)
            new_gamma = gamma[:, index[0]]
            Emission[:, i] = np.sum(new_gamma, axis=1)

        Emission = np.divide(Emission, denominator.reshape((-1, 1)))

        print(n + 1, '\n', Transition, '\n', Emission)

    return Transition, Emission
