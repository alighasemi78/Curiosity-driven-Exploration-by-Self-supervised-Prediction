from itertools import chain

import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torch.utils.data import DataLoader

from agents.agent import Agent


class PPOLoss(_Loss):
    def __init__(
        self,
        clip_range,
        v_clip_range,
        c_entropy,
        c_value,
        reporter,
    ):
        super().__init__()
        self.clip_range = clip_range
        self.v_clip_range = v_clip_range
        self.c_entropy = c_entropy
        self.c_value = c_value
        self.reporter = reporter

    def forward(
        self,
        distribution_old,
        value_old,
        distribution,
        value,
        action,
        reward,
        advantage,
    ):
        # Value loss
        value_old_clipped = value_old + (value - value_old).clamp(
            -self.v_clip_range, self.v_clip_range
        )
        v_old_loss_clipped = (reward - value_old_clipped).pow(2)
        v_loss = (reward - value).pow(2)
        value_loss = torch.min(v_old_loss_clipped, v_loss).mean()

        # Policy loss
        advantage = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-8
        )
        advantage = advantage.detach()
        log_prob = distribution.log_prob(action)
        log_prob_old = distribution_old.log_prob(action)
        ratio = (log_prob - log_prob_old).exp().view(-1)

        surrogate = advantage * ratio
        surrogate_clipped = advantage * ratio.clamp(
            1 - self.clip_range, 1 + self.clip_range
        )
        policy_loss = torch.min(surrogate, surrogate_clipped).mean()

        # Entropy
        entropy = distribution.entropy().mean()

        # Total loss
        losses = policy_loss + self.c_entropy * entropy - self.c_value * value_loss
        total_loss = -losses
        self.reporter.scalar("ppo_loss/policy", -policy_loss.item())
        self.reporter.scalar("ppo_loss/entropy", -entropy.item())
        self.reporter.scalar("ppo_loss/value_loss", value_loss.item())
        self.reporter.scalar("ppo_loss/total", total_loss)
        return total_loss


class PPO(Agent):
    def __init__(
        self,
        env,
        model_factory,
        curiosity_factory,
        reward,
        advantage,
        learning_rate,
        clip_range,
        v_clip_range,
        c_entropy,
        c_value,
        n_mini_batches,
        n_optimization_epochs,
        clip_grad_norm,
        normalize_state,
        normalize_reward,
        reporter,
    ):
        super().__init__(
            env,
            model_factory,
            curiosity_factory,
            normalize_state,
            normalize_reward,
            reporter,
        )
        self.reward = reward
        self.advantage = advantage
        self.n_mini_batches = n_mini_batches
        self.n_optimization_epochs = n_optimization_epochs
        self.clip_grad_norm = clip_grad_norm
        self.optimizer = Adam(
            chain(self.model.parameters(), self.curiosity.parameters()), learning_rate
        )
        self.loss = PPOLoss(clip_range, v_clip_range, c_entropy, c_value, reporter)

    def _train(
        self,
        states,
        actions,
        rewards,
        dones,
    ):
        policy_old, values_old = self.model(
            self._to_tensor(self.state_converter.reshape_as_input(states))
        )
        policy_old = policy_old.detach().view(*states.shape[:2], -1)
        values_old = values_old.detach().view(*states.shape[:2])
        values_old_numpy = values_old.cpu().detach().numpy()
        discounted_rewards = self.reward.discounted(rewards, values_old_numpy, dones)
        advantages = self.advantage.discounted(rewards, values_old_numpy, dones)
        dataset = self.model.dataset(
            policy_old[:, :-1],
            values_old[:, :-1],
            states[:, :-1],
            states[:, 1:],
            actions,
            discounted_rewards,
            advantages,
        )
        loader = DataLoader(
            dataset, batch_size=len(dataset) // self.n_mini_batches, shuffle=True
        )
        with torch.autograd.detect_anomaly():
            for _ in range(self.n_optimization_epochs):
                for tuple_of_batches in loader:
                    (
                        batch_policy_old,
                        batch_values_old,
                        batch_states,
                        batch_next_states,
                        batch_actions,
                        batch_rewards,
                        batch_advantages,
                    ) = self._tensors_to_device(*tuple_of_batches)
                    batch_policy, batch_values = self.model(batch_states)
                    batch_values = batch_values.squeeze()
                    distribution_old = self.action_converter.distribution(
                        batch_policy_old
                    )
                    distribution = self.action_converter.distribution(batch_policy)
                    loss = self.loss(
                        distribution_old,
                        batch_values_old,
                        distribution,
                        batch_values,
                        batch_actions,
                        batch_rewards,
                        batch_advantages,
                    )
                    loss = self.curiosity.loss(
                        loss, batch_states, batch_next_states, batch_actions
                    )
                    # print('loss:', loss)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                    self.optimizer.step()
