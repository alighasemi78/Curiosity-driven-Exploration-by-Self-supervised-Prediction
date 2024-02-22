from torch import nn

from models.datasets import NonSequentialDataset
from models.model import Model, ModelFactory


class MLP(Model):
    def __init__(self, state_space, action_space):
        assert len(state_space.shape) == 1, "Only flat spaces supported by MLP model"
        assert (
            len(action_space.shape) == 1
        ), "Only flat action spaces supported by MLP model"
        super().__init__(state_space, action_space)
        self.input = nn.Sequential(
            nn.Linear(state_space.shape[0], 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.policy_out = self.action_space.policy_out_model(64)
        self.value_out = nn.Linear(64, 1)

    def forward(self, state):
        x = self.input(state)
        policy = self.policy_out(x)
        value = self.value_out(x)
        return policy, value

    def value(self, states):
        _, value = self(states)
        return value

    def policy_logits(self, states):
        policy, _ = self(states)
        return policy

    def dataset(self, *arrays):
        return NonSequentialDataset(*arrays)

    @staticmethod
    def factory():
        return MLPFactory()


class MLPFactory(ModelFactory):
    def create(self, state_space, action_space):
        return MLP(state_space, action_space)
