"""
Solving Mountain Car Continuous version using DDPG
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.ddpg.agent import Agent
from algorithms.utils import DDPGConfig


N_EPISODES = 150


def train(n_eps, agent, env, max_ep_len=999):
    returns_buffer = np.zeros(n_eps)
    for i in range(n_eps):
        obs = env.reset()
        agent.exploration_noise.reset_states()
        return_ = 0.
        for t in range(max_ep_len):
            env.render()

            act = agent.pick_action(torch.from_numpy(obs.astype('float32')))
            nxt_obs, rew, done, _ = env.step(act)
            # Ignore the "done" signal if it comes from hitting the time horizon (that is, when it's an artificial
            # terminal signal that isn't based on the agent's state)
            done = False if t+1 == max_ep_len else done
            return_ += rew

            agent.replay_memory.store(obs, act, rew, nxt_obs, done)

            obs = nxt_obs

            agent.optimize_models()
            agent.update_targets()

            if done or t+1 == max_ep_len:
                print(f"Episode {i + 1} finished. return: {return_: .2f}")
                returns_buffer[i] = return_
                break

    print("Training Complete")
    env.close()
    # return average returns of previous 50 episodes
    return returns_buffer, np.convolve(returns_buffer, np.ones((50,))/50, mode='valid')  # Moving Average


def test():
    pass


if __name__ == "__main__":
    gym.logger.set_level(40)
    environment = gym.make("MountainCarContinuous-v0")

    config = DDPGConfig(2, 1, 1e-4, 1e-3, .99, 1e-3, 100000, 32)
    ddpg_agent = Agent(config)

    returns, ave_returns = train(N_EPISODES, ddpg_agent, environment)

    fig, ax = plt.subplots()
    ax.set_xlabel("number of episodes")
    ax.set_ylabel("average return")
    ax.plot(returns)
    ax.plot(np.arange(100, N_EPISODES + 1), ave_returns)
    plt.show()
