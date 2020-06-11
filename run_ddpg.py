"""
Solving Mountain Car Continuous version using DDPG
"""

import argparse

import gym
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

from deeprl.ddpg.agent import Agent
from deeprl.ddpg.model import ActorNet
from deeprl.utils import DDPGConfig


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
                print(f"Episode {i+1} finished in {t+1} steps with return: {return_: .2f} . "
                      f"Sigma: {agent.exploration_noise.sigma: .2f}")
                returns_buffer[i] = return_
                break

    print("Training Complete")
    env.close()
    # return average returns of previous 50 episodes
    torch.save(agent.actor.state_dict(), f'deeprl/ddpg/params{n_eps}eps.pt')
    return returns_buffer, np.convolve(returns_buffer, np.ones((50,))/50, mode='valid')  # Moving Average


def test(n_eps, n_eps_model, env, max_ep_len=999):
    model = ActorNet(2, 1)

    try:
        model.load_state_dict(torch.load(f'deeprl/ddpg/params{n_eps_model}eps.pt'))
    except FileNotFoundError:
        print("This model is not trained yet")

    model.eval()

    for i in range(n_eps):
        obs = torch.from_numpy(env.reset().astype('float32'))
        return_ = 0
        for t in range(max_ep_len):
            env.render()

            act = model(obs).detach().numpy()
            nxt_obs, rew, done, _ = env.step(act)
            return_ += rew
            obs = torch.from_numpy(nxt_obs.astype('float32').flatten())

            if done or t + 1 == max_ep_len:
                print(f"Episode {i + 1} finished in {t + 1} steps with return: {return_: .2f}")
                break
    env.close()


def eval_model(n_eps_model):
    model = ActorNet(2, 1)

    try:
        model.load_state_dict(torch.load(f'deeprl/ddpg/params{n_eps_model}eps.pt'))
    except FileNotFoundError:
        print("This model is not trained yet")

    model.eval()

    pos = np.linspace(-1.2, 0.6, dtype=np.float32).reshape(-1, 1)
    vel = np.linspace(-0.07, 0.07, dtype=np.float32).reshape(-1, 1)
    pos, vel = np.meshgrid(pos, vel)
    ss = np.dstack((pos, vel))

    policy = model(torch.from_numpy(ss)).squeeze().detach().numpy()

    fig_eval = plt.figure()
    ax_eval = fig_eval.add_subplot(1, 1, 1, projection='3d')
    ax_eval.set_title("learned policy")
    ax_eval.plot_surface(pos, vel, policy, cmap=cm.coolwarm)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="set to test if you want to test the model", action="store_true")
    parser.add_argument('--eval', help="evaluate action values for whole state space", action="store_true")
    args = parser.parse_args()

    N_EPISODES = 200

    gym.logger.set_level(40)
    environment = gym.make("MountainCarContinuous-v0")

    if args.test:
        test(20, N_EPISODES, environment)
    elif args.eval:
        eval_model(N_EPISODES)
    else:
        config = DDPGConfig(n_states=2,
                            n_actions=1,
                            pi_lr=1e-4,
                            q_lr=1e-3,
                            gamma=.99,
                            tau=1e-3,
                            memory_size=100000,
                            batch_size=32,
                            sigma_f=.01,
                            n_steps_annealing=30000)

        ddpg_agent = Agent(config)

        returns, ave_returns = train(N_EPISODES, ddpg_agent, environment)

        fig, ax = plt.subplots()
        ax.set_xlabel("number of episodes")
        ax.set_ylabel("average return")
        ax.plot(returns)
        ax.plot(np.arange(50, N_EPISODES + 1), ave_returns)
        plt.show()
