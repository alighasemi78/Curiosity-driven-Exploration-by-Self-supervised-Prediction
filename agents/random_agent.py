import numpy as np

from agents.agent import Agent


class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return np.asarray([self.action_space.sample() for _ in state])

    def _train(
        self,
        states,
        actions,
        rewards,
        dones,
    ):
        pass
