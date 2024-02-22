from rewards.advantage import Advantage
from rewards.n_step_reward import NStepReward


class NStepAdvantage(Advantage):
    def __init__(self, gamma, n=None):
        self.n_step_reward = NStepReward(gamma, n)

    def discounted(self, rewards, values, dones):
        return self.n_step_reward.discounted(rewards, values, dones) - values[:, :-1]
