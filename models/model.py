from abc import ABCMeta, abstractmethod

from torch import nn


class Model(nn.Module, metaclass=ABCMeta):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

    @abstractmethod
    def value(self, states):
        raise NotImplementedError("Implement me")

    @abstractmethod
    def policy_logits(self, states):
        raise NotImplementedError("Implement me")

    @abstractmethod
    def dataset(self, *arrays):
        raise NotImplementedError("Implement me")

    @staticmethod
    @abstractmethod
    def factory():
        raise NotImplementedError("Implement me")


class ModelFactory:
    @abstractmethod
    def create(self, state_space, action_space):
        raise NotImplementedError("Implement me")
