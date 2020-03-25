"""
Solving Mountain Car Continuous using Deep Deterministic Policy Gradient
"""
from collections import namedtuple

import numpy as np
import torch
import torch.optim as optim

from .model import ActorNet, CriticNet
from ..utils import ReplayMemory, OrnsteinUhlenbeckProcess


class Agent:
    """
    abstract class for actor and critic agents
    """
    def __init__(self, config: namedtuple):
        self.n_states = config.n_states
        self.n_actions = config.n_actions

        self.critic = CriticNet(self.n_states, self.n_actions)
        self.critic_target = CriticNet(self.n_states, self.n_actions)
        self.critic_target.load_state_dict(self.critic.state_dict())  # equalize initial parameters
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.q_lr)

        self.actor = ActorNet(self.n_states, self.n_actions)
        self.actor_target = ActorNet(self.n_states, self.n_actions)
        self.actor_target.load_state_dict(self.actor.state_dict())  # equalize initial parameters
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.pi_lr)

        self.replay_memory = ReplayMemory(self.n_states, self.n_actions, config.memory_size)
        self.exploration_noise = OrnsteinUhlenbeckProcess(sigma_i=1., sigma_f=config.sigma_f,
                                                          n_steps_annealing=config.n_steps_annealing)

        self.gamma = config.gamma
        self.tau = config.tau
        self.batch_size = config.batch_size

        self.q_criterion = torch.nn.MSELoss()

    def pick_action(self, observation: torch.tensor) -> np.array:
        """
        Pick actions according to policy network and exploration noise
        """
        action = self.actor(observation).detach().numpy()
        noise = self.exploration_noise.sample()
        return np.clip(action+noise, -1, 1)

    def optimize_models(self):
        if len(self.replay_memory) < self.batch_size:
            return

        batch = self.replay_memory.sample_batch(self.batch_size)

        # Optimize critic one step
        self.critic_optimizer.zero_grad()
        q_loss = self._compute_q_loss(batch)
        q_loss.backward()
        self.critic_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.critic.parameters():
            p.requires_grad = False
        # Optimize actor one step
        self.actor_optimizer.zero_grad()
        pi_loss = self._compute_pi_loss(batch)
        pi_loss.backward()
        self.actor_optimizer.step()
        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.critic.parameters():
            p.requires_grad = True

    def _compute_q_loss(self, data):
        obs, act, rew, nxt_obs, done = data['obs'], data['act'], data['rew'], data['nxt_obs'], data['done']
        with torch.no_grad():
            nxt_targ_act = self.actor_target(nxt_obs)
            td_target = rew + self.gamma * (1 - done) * self.critic_target(nxt_obs, nxt_targ_act)
        cur_q = self.critic(obs, act)
        loss = self.q_criterion(cur_q, td_target)
        return loss

    def _compute_pi_loss(self, data):
        obs = data['obs']
        q_pi = self.critic(obs, self.actor(obs))
        return -q_pi.mean()

    def update_targets(self):
        """
        Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilize training
        """
        model_pairs = {(self.critic, self.critic_target), (self.actor, self.actor_target)}
        with torch.no_grad():
            for pair in model_pairs:
                for p, p_target in zip(pair[0].parameters(), pair[1].parameters()):
                    p_target.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_target.data)
