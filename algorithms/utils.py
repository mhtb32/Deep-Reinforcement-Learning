"""
Replay memory for storing experience samples
"""

from collections import namedtuple
import random

import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory:
    """
    Stores past experiences of agent as (s,a,r,s') tuples
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """
        Saves a transition.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class OrnsteinUhlenbeckProcess:
    """
    Implementation based on:
        https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        https://github.com/ghliu/pytorch-ddpg/blob/master/random_process.py#L32
    """
    def __init__(self, theta=1., mu=0., sigma_i=1., sigma_f=None, dt=1e-2, x0=None, size=1, n_steps_annealing=1000):
        self.theta = theta
        self.mu = mu
        self.sigma_i = sigma_i
        self.dt = dt

        self.size = size

        self.x0 = x0
        self.xt = x0 if x0 is not None else np.zeros(self.size)

        self.n_steps = 0

        if sigma_f is None:
            self.m = 0
            self.sigma_f = sigma_i
        else:
            self.m = -float(sigma_i - sigma_f) / n_steps_annealing
            self.sigma_f = sigma_f

    @property
    def sigma(self):
        return max(self.sigma_f, self.sigma_i + self.m * float(self.n_steps))

    def sample(self):
        xt = self.xt + self.theta * (self.mu - self.xt) * self.dt + self.sigma * np.sqrt(self.dt) * \
             np.random.normal(size=self.size)
        self.xt = xt
        self.n_steps += 1
        return xt

    def reset_states(self):
        self.xt = self.x0 if self.x0 is not None else np.zeros(self.size)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DURATION = 10000

    rv = OrnsteinUhlenbeckProcess(mu=3.5)
    x = np.zeros(DURATION)

    for t in range(DURATION):
        x[t] = rv.sample()

    plt.figure()
    plt.plot(x)
    plt.show()
