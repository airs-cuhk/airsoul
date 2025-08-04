import gym
import numpy
from numpy import random

class TabularQ(object):
    """
    Solver for any discrete env with Q-Learning
    """
    def __init__(self, ns, na, gamma=0.99, c=0.01, alpha=0.01, max_steps=4000):
        """
        The constructor for the class AnyMDPSolverQ
        The exploration strategy is controlled by UCB-H with c as its hyperparameter. Increasing c will encourage exploration
        Simulation of the ideal policy when the ground truth is not known
        """
        self.n_actions = na
        self.n_states = ns
        self.value_matrix = numpy.zeros((self.n_states, self.n_actions))
        self.sa_vistied = numpy.zeros((self.n_states, self.n_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.max_steps = max_steps
        self._c = c / (1.0 - self.gamma)

    def learner(self, s, a, ns, r, done):
        if(done):
            target = r
        else:
            target = r + self.gamma * max(self.value_matrix[ns])
        error = target - self.value_matrix[s][a]
        self.value_matrix[s][a] += self.alpha * error
        self.sa_vistied[s][a] += 1

    def policy(self, state):
        # Apply UCB with dynamic noise (Thompson Sampling)
        values = self._c * numpy.sqrt(numpy.log(self.max_steps + 1) / numpy.clip(self.sa_vistied[state], 1.0, None)) * \
                numpy.maximum(numpy.random.randn(self.n_actions), 0) + \
                self.value_matrix[state]
        return int(numpy.argmax(values))