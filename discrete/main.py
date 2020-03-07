"""
Train/Test RL agent using DQN as action-value function approximator
"""

import gym
import torch
import numpy as np
import itertools
from discrete.dqn import Agent
from collections import deque
import matplotlib.pyplot as plt


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
            reward += 100/7 * np.fabs(next_state[1])  # shaping reward with normalized speed
            mod_e_return += reward

            reward = torch.tensor([reward], device=agent.device)
            next_state = torch.from_numpy(next_state.astype('float32')).to(agent.device)

            if done:
                print(f"Episode {i_episode + 1} with return {e_return} and modified return {mod_e_return}")
                returns_buffer.append(e_return)
                break

            agent.memory.push(state, action, reward, next_state)

            state = next_state

            agent.optimize_model()

        if i_episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print("Training Complete")
    env.close()
    # return average returns of previous 100 episodes
    return [ave for ave in moving_average(returns_buffer, 100)]


def test(n_episodes, agent, env):
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    environment = gym.make('MountainCar-v0')
    car_agent = Agent(2, 3, device=device)

    ave_return = train(500, car_agent, environment)

    plt.figure()
    plt.xlabel("number of episodes")
    plt.ylabel("average return")
    plt.plot(ave_return)
    plt.show()
