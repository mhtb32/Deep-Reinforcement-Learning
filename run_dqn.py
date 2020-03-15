"""
Solve Discrete Mountain Car Problem using Deep Q Network
"""

import argparse

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.dqn.agent import Agent
from algorithms.dqn.agent import train, test, eval_model


N_EPISODES = 300


parser = argparse.ArgumentParser()
parser.add_argument('--test', help="set to test if you want to test the model", action="store_true")
parser.add_argument('--eval', help="evaluate action values for whole state space", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

environment = gym.make('MountainCar-v0')

if args.test:
    test(20, N_EPISODES, environment, device)
elif args.eval:
    eval_model(N_EPISODES)
else:
    car_agent = Agent(2, 3, device=device)
    returns, ave_returns = train(N_EPISODES, car_agent, environment)

    fig, ax = plt.subplots()
    ax.set_xlabel("number of episodes")
    ax.set_ylabel("average return")
    ax.plot(returns)
    ax.plot(np.arange(100, N_EPISODES + 1), ave_returns)
    plt.show()
