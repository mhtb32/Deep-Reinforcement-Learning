"""
Solving Mountain Car Continuous using Deep Deterministic Policy Gradient
"""

from abc import ABC, abstractmethod


class Agent(ABC):
    """
    abstract class for actor and critic agents
    """
    def __init__(self):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def optimize(self):
        pass


class Actor(Agent):
    """
    Actor as suggested in DDPG
    """
    def __init__(self):
        super().__init__()

    def update_target(self):
        pass

    def optimize(self):
        pass


class Critic(Agent):
    """
    Critic as suggested in DDPG
    """
    def __init__(self):
        super().__init__()

    def update_target(self):
        pass

    def optimize(self):
        pass
