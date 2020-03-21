"""
models for function approximator
"""

import torch
import torch.nn as nn


class ActorNet(nn.Module):
    """
    A MLP with two hidden layers
    inputs = array of states
    output = value of action(s)
    """
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.hl1 = nn.Linear(n_states, 32)
        self.hl2 = nn.Linear(32, 64)
        self.ol = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.hl1(x))
        x = torch.relu(self.hl2(x))
        x = torch.tanh(self.ol(x))  # to scale output between (-1, 1)
        return x


class CriticNet(nn.Module):
    """
    A MLP with two hidden layer
    inputs = array of states
    outputs = estimated q value
    """
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.hl1 = nn.Linear(n_states+n_actions, 32)
        self.hl2 = nn.Linear(32, 64)
        self.ol = nn.Linear(64, 1)

    def forward(self, x, a):
        xa = torch.cat((x, a), dim=-1)
        out = torch.relu(self.hl1(xa))
        out = torch.relu(self.hl2(out))
        out = self.ol(out)
        return torch.squeeze(out, -1)


if __name__ == "__main__":
    actor = ActorNet(2, 1)
    critic = CriticNet(2, 1)

    x_s = torch.tensor([.5, 1.2])
    a_s = torch.tensor([.7])
    print(actor(x_s))
    print(critic(x_s, a_s))
