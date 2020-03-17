"""
Describes agent
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import Net
from ..utils import Transition, ReplayMemory


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
        self.exploration_samples = 4000
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


def reward_shape(state, mode='position'):
    if mode == 'position':  # encourage going above a certain height
        checkpoint = -0.3
        if state[0] > checkpoint:
            return 1.0
        else:
            return 0.0
    elif mode == 'velocity':  # encourage going faster
        return 100/7 * np.fabs(state[1])
    elif mode == 'extra':  # give extra reward for reaching goal
        if state[0] >= 0.5:
            return 10.0
        else:
            return 0.0


def train(n_episodes, agent, env):
    returns_buffer = np.zeros(n_episodes)
    for i in range(n_episodes):
        state = torch.from_numpy(env.reset().astype('float32')).to(agent.device)
        return_ = 0
        mod_return = 0
        for t in range(200):  # 200 is max time-step
            env.render()

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.item())

            return_ += reward
            reward += reward_shape(next_state, mode='position')
            mod_return += reward

            reward = torch.tensor([reward], device=agent.device)
            next_state = torch.from_numpy(next_state.astype('float32')).to(agent.device)

            if done:
                print(f"Episode {i + 1} finished. return: {return_}, mod_return: {mod_return: .2f},"
                      f" epsilon: {agent.eps: .2f}")
                returns_buffer[i] = return_
                break

            agent.memory.push(state, action, reward, next_state)

            state = next_state

            agent.optimize_model()

        if i % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print("Training Complete")
    env.close()
    torch.save(agent.policy_net.state_dict(), f'algorithms/dqn/params{n_episodes}eps.pt')
    # return average returns of previous 100 episodes
    return returns_buffer, np.convolve(returns_buffer, np.ones((100,))/100, mode='valid')  # Moving Average


def test(n_episodes, n_eps_model, env, dev=torch.device('cpu')):
    model = Net(2, 3)
    try:
        model.load_state_dict(torch.load(f'algorithms/dqn/params{n_eps_model}eps.pt'))
    except FileNotFoundError:
        print("Model is not trained yet")
    model.eval()

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


def eval_model(n_episodes: int):
    model = Net(2, 3)

    try:
        model.load_state_dict(torch.load(f'algorithms/dqn/params{n_episodes}eps.pt'))
    except FileNotFoundError:
        print("Model is not trained yet")
    model.eval()

    pos = np.linspace(-1.2, 0.6, dtype=np.float32).reshape(-1, 1)
    vel = np.linspace(-0.07, 0.07, dtype=np.float32).reshape(-1, 1)
    pos, vel = np.meshgrid(pos, vel)
    ss = np.dstack((pos, vel))
    q_values = model(torch.from_numpy(ss)).detach().numpy()

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    ax1.plot_surface(pos, vel, -np.max(q_values, axis=2), cmap=cm.coolwarm)
    ax2.plot_surface(pos, vel, np.argmax(q_values, axis=2), cmap=cm.coolwarm)

    plt.show()
