"""
Describes agent
"""

import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import itertools
from .model import Net
from algorithms.utils import Transition, ReplayMemory


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


def moving_average(iterable, n=3):
    # moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
    # http://en.wikipedia.org/wiki/Moving_average
    it = iter(iterable)
    d = deque(itertools.islice(it, n-1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / n


def reward_shape(state, mode='velocity'):
    if mode == 'position':
        if state[0] > -0.2:
            return 1.0
        else:
            return 0.0
    elif mode == 'velocity':
        return 100/7 * np.fabs(state[1])


def train(n_episodes, agent, env):
    returns_buffer = []
    for i_episode in range(n_episodes):
        state = torch.from_numpy(env.reset().astype('float32')).to(agent.device)
        e_return = 0
        mod_e_return = 0
        for t in range(200):  # 200 is maximum length of an episode
            env.render()

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.item())

            e_return += reward
            reward += reward_shape(next_state, mode='position')
            mod_e_return += reward

            reward = torch.tensor([reward], device=agent.device)
            next_state = torch.from_numpy(next_state.astype('float32')).to(agent.device)

            if done:
                print(f"Episode {i_episode + 1} with return {e_return} and modified return {mod_e_return}.")
                returns_buffer.append(e_return)
                break

            agent.memory.push(state, action, reward, next_state)

            state = next_state

            agent.optimize_model()

        if i_episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print("Training Complete")
    env.close()
    torch.save(agent.target_net.state_dict(), 'param.pt')
    # return average returns of previous 100 episodes
    return [ave for ave in moving_average(returns_buffer, 100)]


def test(n_episodes, env, dev=torch.device('cpu')):
    model = Net(2, 3)
    try:
        model.load_state_dict(torch.load('param.pt'))
    except FileNotFoundError:
        print("Model is not trained yet")
    for i in range(n_episodes):
        state = torch.from_numpy(env.reset().astype('float32')).to(dev)
        e_return = 0
        for t in range(200):
            env.render()
            action = model(state).argmax().view(1, 1)
            next_state, reward, done, _ = env.step(action.item())
            e_return += reward
            if done:
                print(f"Episode {i + 1} reward is {e_return}")
                break
