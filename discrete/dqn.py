"""
Describes agent
"""

import random
import math
from collections import namedtuple
from discrete.model import Net

import torch
import torch.nn as nn
import torch.optim as optim

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


class Agent:
    def __init__(self, n_states, n_actions, lr=1e-3, gamma=0.99, memory_size=20000, batch_size=32,  device='cpu'):
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = device
        self.policy_net = Net(n_states, n_actions).to(self.device)
        self.target_net = Net(n_states, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # equalize parameters
        self.target_net.eval()  # switch training mode off
        self.memory = ReplayMemory(memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.step_count = 0  # to count steps of training

        # Hyper Parameters
        self.eps_initial = 1  # mainly explore at first
        self.eps_final = 0.01  # leave a bit of exploration
        self.eps_decay = 200  # decay epsilon every 200 steps
        self.eps = self.eps_initial
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = 1  # in number of episodes

        self.criterion = nn.MSELoss()

    def select_action(self, state):
        """
        following an e-greedy policy

        :param state: current state
        :return: selected action from {0, 1, 2}
        """
        sample = random.random()
        # Decay epsilon every "eps_decay" steps to make policy GLIE (see David Silver's course)
        self.eps = self.eps_final + (self.eps_initial - self.eps_final) * math.exp(-1. * self.step_count /
                                                                                   self.eps_decay)
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
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).view(-1, self.n_states)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor([True if s is not None else False for s in batch.next_state],
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).view(-1, self.n_states)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # Q(s_t, a)
        # see https://stackoverflow.com/q/50999977/8122011 for more information

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch  # TD-target

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
