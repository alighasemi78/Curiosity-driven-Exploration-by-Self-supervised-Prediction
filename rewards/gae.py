import numpy as np

from rewards.advantage import Advantage
from rewards.utils import discount


class GeneralizedAdvantageEstimation(Advantage):
    def __init__(self, gamma, lam):
        self.gamma = gamma
        self.lam = lam

    def discounted(self, rewards, values, dones):
        td_errors = (
            rewards + self.gamma * values[:, 1:] * (1.0 - dones) - values[:, :-1]
        )
        return discount(
            td_errors, np.zeros_like(values[:, 0]), dones, self.lam * self.gamma
        )
