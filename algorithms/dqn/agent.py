"""
Describes agent
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import Net
from ..utils import ReplayMemory


class Agent:
    def __init__(self, n_states, n_actions, lr=1e-3, gamma=0.99, memory_size=20000, batch_size=32, target_update=1,
                 device='cpu'):
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = device
        self.policy_net = Net(n_states, n_actions).to(self.device)
        self.target_net = Net(n_states, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # equalize parameters
        self.target_net.eval()  # switch training mode off
        self.memory = ReplayMemory(n_states, 1, memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.step_count = 0  # to count steps of training

        # Hyper Parameters
        self.eps_initial = 1  # mainly explore at first
        self.eps_final = 0.01  # leave a bit of exploration
        self.exploration_samples = 4000
        self.eps = self.eps_initial
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update  # in number of episodes

        self.criterion = nn.MSELoss()

    def select_action(self, state):
        """
        following an e-greedy policy

        :param state: current state
        :return: selected action from {0, 1, 2}
        """
        sample = np.random.rand()
        # Decay epsilon every "eps_decay" steps to make policy GLIE (see David Silver's course)
        fraction = min(float(self.step_count) / self.exploration_samples, 1.0)
        self.eps = self.eps_initial + fraction * (self.eps_final - self.eps_initial)
        self.step_count += 1
        if sample > self.eps:
            with torch.no_grad():  # argmax() should not affect gradient
                return self.policy_net(state).argmax().view(1, 1)
        else:
            return torch.randint(self.n_actions, (1, 1), device=self.device)  # Uniformly distributed random number
            # from {0, 1, 2}

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample_batch(self.batch_size)

        state_action_values = self.policy_net(batch['obs']).gather(1, batch['act'].long())  # Q(s_t, a), see
        # https://stackoverflow.com/q/50999977/8122011 for more information.

        with torch.no_grad():
            next_state_values = self.target_net(batch['nxt_obs']).max(1)[0]
            # TD-target:
            target_state_action_values = batch['rew'] + self.gamma * (1 - batch['done']) * next_state_values

        loss = self.criterion(state_action_values, target_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
