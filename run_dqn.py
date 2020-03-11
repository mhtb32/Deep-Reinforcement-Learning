"""
Solve Discrete Mountain Car Problem using Deep Q Network
"""

import argparse
import torch
import gym
import matplotlib.pyplot as plt
from algorithms.dqn.agent import Agent
from algorithms.dqn.agent import train, test


parser = argparse.ArgumentParser()
parser.add_argument('--test', help="set to test if you want to test the model", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

environment = gym.make('MountainCar-v0')

if not args.test:
    car_agent = Agent(2, 3, device=device)
    ave_return = train(500, car_agent, environment)

    plt.figure()
    plt.xlabel("number of episodes")
    plt.ylabel("average return")
    plt.plot(ave_return)
    plt.show()
else:
    test(20, environment, device)
