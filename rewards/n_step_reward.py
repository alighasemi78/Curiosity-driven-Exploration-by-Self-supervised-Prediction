from rewards.reward import Reward
from rewards.utils import discount
import numpy as np


class NStepReward(Reward):
    def __init__(self, gamma, n=None):
        self.gamma = gamma
        self.n = n

    def discounted(self, rewards, values, dones):
        if self.n is None:
            return discount(rewards, values[:, -1], dones, self.gamma)
        discounted = np.zeros_like(rewards)
        for start in range(rewards.shape[1]):
            end = min(start + self.n, rewards.shape[1])
            discounted[:, start] = discount(
                rewards[:, start:end], values[:, end], dones[:, start:end], self.gamma
            )[:, 0]
        return discounted
