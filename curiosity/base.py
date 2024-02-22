from abc import abstractmethod, ABCMeta

import torch


class Curiosity(metaclass=ABCMeta):
    def __init__(self, state_converter, action_converter):
        self.state_converter = state_converter
        self.action_converter = action_converter
        self.device = None
        self.dtype = None

    @abstractmethod
    def reward(self, rewards, states, actions):
        raise NotImplementedError("Implement me")

    @abstractmethod
    def loss(self, policy_loss, states, next_states, actions):
        raise NotImplementedError("Implement me")

    @abstractmethod
    def parameters(self):
        raise NotImplementedError("Implement me")

    def to(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def _to_tensors(self, *arrays):
        return [
            torch.tensor(array, device=self.device, dtype=self.dtype)
            for array in arrays
        ]


class CuriosityFactory(metaclass=ABCMeta):
    @abstractmethod
    def create(self, state_converter, action_converter):
        raise NotImplementedError("Implement me")
