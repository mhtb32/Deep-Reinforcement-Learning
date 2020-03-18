"""
Replay memory for storing experience samples
"""

import numpy as np
import torch


def combined_shape(length, shape=None):
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayMemory:
    """
    A simple FIFO experience replay buffer.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.nxt_obs = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def __len__(self):
        return self.size

    def store(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.nxt_obs[self.ptr] = next_obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs[idxs],
                     nxt_obs=self.nxt_obs[idxs],
                     act=self.act[idxs],
                     rew=self.rew[idxs],
                     done=self.done[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


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
