from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from curiosity.base import Curiosity, CuriosityFactory


class ICMModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, state_converter, action_converter):
        super().__init__()
        self.state_converter = state_converter
        self.action_converter = action_converter

    @staticmethod
    @abstractmethod
    def factory():
        raise NotImplementedError("Implement me")


class ICMModelFactory:
    def create(self, state_converter, action_converter):
        raise NotImplementedError("Implement me")


class ForwardModel(nn.Module):
    def __init__(self, action_converter, state_latent_features):
        super().__init__()
        self.action_converter = action_converter
        action_latent_features = 128
        if action_converter.discrete:
            self.action_encoder = nn.Embedding(
                action_converter.shape[0], action_latent_features
            )
        else:
            self.action_encoder = nn.Linear(
                action_converter.shape[0], action_latent_features
            )
        self.hidden = nn.Sequential(
            nn.Linear(action_latent_features + state_latent_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, state_latent_features),
        )

    def forward(self, state_latent, action):
        action = self.action_encoder(
            action.long() if self.action_converter.discrete else action
        )
        x = torch.cat((action, state_latent), dim=-1)
        x = self.hidden(x)
        return x


class InverseModel(nn.Module):
    def __init__(self, action_converter, state_latent_features):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(state_latent_features * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            action_converter.policy_out_model(128),
        )

    def forward(self, state_latent, next_state_latent):
        return self.input(torch.cat((state_latent, next_state_latent), dim=-1))


class MlpICMModel(ICMModel):
    def __init__(self, state_converter, action_converter):
        assert (
            len(state_converter.shape) == 1
        ), "Only flat spaces supported by MLP model"
        assert (
            len(action_converter.shape) == 1
        ), "Only flat action spaces supported by MLP model"
        super().__init__(state_converter, action_converter)
        self.encoder = nn.Sequential(
            nn.Linear(state_converter.shape[0], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )
        self.forward_model = ForwardModel(action_converter, 128)
        self.inverse_model = InverseModel(action_converter, 128)

    def forward(self, state, next_state, action):
        state = self.encoder(state)
        next_state = self.encoder(next_state)
        next_state_hat = self.forward_model(state, action)
        action_hat = self.inverse_model(state, next_state)
        return next_state, next_state_hat, action_hat

    @staticmethod
    def factory():
        return MlpICMModelFactory()


class MlpICMModelFactory(ICMModelFactory):
    def create(self, state_converter, action_converter):
        return MlpICMModel(state_converter, action_converter)


class ICM(Curiosity):
    def __init__(
        self,
        state_converter,
        action_converter,
        model_factory,
        policy_weight,
        reward_scale,
        weight,
        intrinsic_reward_integration,
        reporter,
    ):
        super().__init__(state_converter, action_converter)
        self.model = model_factory.create(state_converter, action_converter)
        self.policy_weight = policy_weight
        self.reward_scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter

    def parameters(self):
        return self.model.parameters()

    def reward(self, rewards, states, actions):
        n, t = actions.shape[0], actions.shape[1]
        states, next_states = states[:, :-1], states[:, 1:]
        states, next_states, actions = self._to_tensors(
            self.state_converter.reshape_as_input(states),
            self.state_converter.reshape_as_input(next_states),
            actions.reshape(n * t, *actions.shape[2:]),
        )
        next_states_latent, next_states_hat, _ = self.model(
            states, next_states, actions
        )
        intrinsic_reward = (
            self.reward_scale
            / 2
            * (next_states_hat - next_states_latent).norm(2, dim=-1).pow(2)
        )
        intrinsic_reward = intrinsic_reward.cpu().detach().numpy().reshape(n, t)
        self.reporter.scalar(
            "icm/reward",
            intrinsic_reward.mean().item()
            if self.reporter.will_report("icm/reward")
            else 0,
        )
        return (
            1.0 - self.intrinsic_reward_integration
        ) * rewards + self.intrinsic_reward_integration * intrinsic_reward

    def loss(self, policy_loss, states, next_states, actions):
        next_states_latent, next_states_hat, actions_hat = self.model(
            states, next_states, actions
        )
        forward_loss = (
            0.5
            * (next_states_hat - next_states_latent.detach())
            .norm(2, dim=-1)
            .pow(2)
            .mean()
        )
        inverse_loss = self.action_converter.distance(actions_hat, actions)
        curiosity_loss = self.weight * forward_loss + (1 - self.weight) * inverse_loss
        self.reporter.scalar("icm/loss", curiosity_loss.item())
        return self.policy_weight * policy_loss + curiosity_loss

    def to(self, device, dtype):
        super().to(device, dtype)
        self.model.to(device, dtype)

    @staticmethod
    def factory(
        model_factory,
        policy_weight,
        reward_scale,
        weight,
        intrinsic_reward_integration,
        reporter,
    ):
        return ICMFactory(
            model_factory,
            policy_weight,
            reward_scale,
            weight,
            intrinsic_reward_integration,
            reporter,
        )


class ICMFactory(CuriosityFactory):
    def __init__(
        self,
        model_factory,
        policy_weight,
        reward_scale,
        weight,
        intrinsic_reward_integration,
        reporter,
    ):
        self.policy_weight = policy_weight
        self.model_factory = model_factory
        self.scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter

    def create(self, state_converter, action_converter):
        return ICM(
            state_converter,
            action_converter,
            self.model_factory,
            self.policy_weight,
            self.scale,
            self.weight,
            self.intrinsic_reward_integration,
            self.reporter,
        )
