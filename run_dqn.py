"""
Solve Discrete Mountain Car Problem using Deep Q Network
"""

import argparse

import gym
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

from algorithms.dqn.agent import Agent
from algorithms.dqn.model import Net

N_EPISODES = 400


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


def train(n_episodes, agent, env, max_ep_len=200):
    returns_buffer = np.zeros(n_episodes)
    for i in range(n_episodes):
        observation = env.reset()
        return_, mod_return = 0., 0.
        for t in range(max_ep_len):
            env.render()

            action = agent.pick_action(torch.as_tensor(observation.astype('float32'), device=agent.device))
            next_observation, reward, done, _ = env.step(action.item())

            # Ignore the "done" signal if it comes from hitting the time horizon (that is, when it's an artificial
            # terminal signal that isn't based on the agent's state)
            done = False if t+1 == max_ep_len else done
            return_ += reward
            reward += reward_shape(next_observation, mode='position')
            mod_return += reward
            agent.memory.store(observation, action, reward, next_observation, done)

            observation = next_observation

            agent.optimize_model()

            if done or t+1 == max_ep_len:
                print(f"Episode {i + 1} finished. return: {return_}, mod_return: {mod_return: .2f},"
                      f" epsilon: {agent.eps: .2f}")
                returns_buffer[i] = return_
                break

        if i % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print("Training Complete")
    env.close()
    torch.save(agent.policy_net.state_dict(), f'algorithms/dqn/params{n_episodes}eps.pt')
    # return average returns of previous 100 episodes
    return returns_buffer, np.convolve(returns_buffer, np.ones((100,))/100, mode='valid')  # Moving Average


def test(n_episodes, n_eps_model, env):
    model = Net(2, 3)
    try:
        model.load_state_dict(torch.load(f'algorithms/dqn/params{n_eps_model}eps.pt'))
    except FileNotFoundError:
        print("Model is not trained yet")
    model.eval()

    for i in range(n_episodes):
        state = torch.from_numpy(env.reset().astype('float32'))
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

    fig_eval = plt.figure()
    ax1 = fig_eval.add_subplot(2, 1, 1, projection='3d')
    ax1.set_title("cost to go")
    ax2 = fig_eval.add_subplot(2, 1, 2, projection='3d')
    ax2.set_title("policy")
    ax1.plot_surface(pos, vel, -np.max(q_values, axis=2), cmap=cm.coolwarm)
    ax2.plot_surface(pos, vel, np.argmax(q_values, axis=2), cmap=cm.coolwarm)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="set to test if you want to test the model", action="store_true")
    parser.add_argument('--eval', help="evaluate action values for whole state space", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    environment = gym.make('MountainCar-v0')
    # environment = environment.unwrapped

    if args.test:
        test(20, N_EPISODES, environment)
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
