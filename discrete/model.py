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
        self.hidden_layer = nn.Linear(n_states, 64)
        self.output_layer = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.sigmoid(self.hidden_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


if __name__ == "__main__":
    net = Net(2, 3)
    print(net)
