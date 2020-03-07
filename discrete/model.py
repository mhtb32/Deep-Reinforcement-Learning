"""
contains description of approximator model
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """
    A MLP with one hidden layer
    inputs = array of states
    outputs = estimated q value for each action
    """
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.hl1 = nn.Linear(n_states, 24)
        self.hl2 = nn.Linear(24, 48)
        self.ol = nn.Linear(48, n_actions)

    def forward(self, x):
        x = torch.relu(self.hl1(x))
        x = torch.relu(self.hl2(x))
        x = self.ol(x)
        return x


if __name__ == "__main__":
    net = Net(2, 3)
    print(net)
