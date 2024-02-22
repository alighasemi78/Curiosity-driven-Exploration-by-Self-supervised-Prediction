from abc import abstractmethod, ABCMeta

import gym.spaces as spaces
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import CrossEntropyLoss, MSELoss

from normalizers import StandardNormalizer, NoNormalizer


class Converter(metaclass=ABCMeta):
    @property
    @abstractmethod
    def discrete(self):
        raise NotImplementedError("Implement me")

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError("Implement me")

    @abstractmethod
    def distribution(self, logits):
        raise NotImplementedError("Implement me")

    @abstractmethod
    def reshape_as_input(self, array):
        raise NotImplementedError("Implement me")

    @abstractmethod
    def action(self, tensor):
        raise NotImplementedError("Implement me")

    @abstractmethod
    def distance(self, policy_logits, y):
        raise NotImplementedError("Implement me")

    @abstractmethod
    def state_normalizer(self):
        raise NotImplementedError("Implement me")

    @abstractmethod
    def policy_out_model(self, in_features):
        raise NotImplementedError("Implement me")

    @staticmethod
    def for_space(space):
        if isinstance(space, spaces.Discrete):
            return DiscreteConverter(space)
        elif isinstance(space, spaces.Box):
            return BoxConverter(space)


class DiscreteConverter(Converter):
    def __init__(self, space):
        self.space = space
        self.loss = CrossEntropyLoss()

    @property
    def discrete(self):
        return True

    @property
    def shape(self):
        return (self.space.n,)

    def distribution(self, logits):
        return Categorical(logits=logits)

    def reshape_as_input(
        self,
        array,
    ):
        return array.reshape(array.shape[0] * array.shape[1], -1)

    def action(self, tensor):
        return self.distribution(tensor).sample()

    def distance(self, policy_logits, y):
        return self.loss(policy_logits, y.long())

    def state_normalizer(self):
        return NoNormalizer()

    def policy_out_model(self, in_features):
        return nn.Linear(in_features, self.shape[0])


class BoxConverter(Converter):
    def __init__(self, space):
        self.space = space
        self.loss = MSELoss()

    @property
    def discrete(self):
        return False

    @property
    def shape(self):
        return self.space.shape

    def distribution(self, logits):
        assert logits.size(1) % 2 == 0
        mid = logits.size(1) // 2
        loc = logits[:, :mid]
        scale = logits[:, mid:]
        return Normal(loc, scale)

    def reshape_as_input(self, array):
        return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    def action(self, tensor):
        min = torch.tensor(self.space.low, device=tensor.device)
        max = torch.tensor(self.space.high, device=tensor.device)
        return torch.max(torch.min(self.distribution(logits=tensor).sample(), max), min)

    def distance(self, policy_logits, y):
        return self.loss(self.action(policy_logits), y)

    def state_normalizer(self):
        return StandardNormalizer()

    def policy_out_model(self, in_features):
        return NormalDistributionModule(in_features, self.shape[0])


class NormalDistributionModule(nn.Module):
    def __init__(self, in_features, n_action_values):
        super().__init__()
        self.policy_mean = nn.Linear(in_features, n_action_values)
        self.policy_std = nn.Parameter(torch.zeros(1, n_action_values))

    def forward(self, x):
        policy = self.policy_mean(x)
        policy_std = self.policy_std.expand_as(policy).exp()
        return torch.cat((policy, policy_std), dim=-1)
