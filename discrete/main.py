"""
Train/Test RL agent using DQN as action-value function approximator
"""

import gym
import torch
from discrete.dqn import Agent


def train(n_episodes, agent, env):
    for i_episode in range(n_episodes):
        state = torch.from_numpy(env.reset().astype('float32')).to(agent.device)
        for t in range(200):  # 200 is maximum length of an episode
            env.render()

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=agent.device)
            next_state = torch.from_numpy(next_state.astype('float32')).to(agent.device)

            if done:
                print(f"Episode finished after {t + 1} time steps")
                break

            agent.memory.push(state, action, reward, next_state)

            state = next_state

            agent.optimize_model()
            print(f"time_step: {t}")

        if i_episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print("Training Complete")
    env.close()


def test(n_episodes, agent, env):
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    environment = gym.make('MountainCar-v0')
    car_agent = Agent(2, 3, device=device)

    train(50, car_agent, environment)
