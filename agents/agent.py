import torch

from envs import Runner, Converter, RandomRunner
from normalizers import StandardNormalizer, NoNormalizer


class Agent:
    def __init__(
        self,
        env,
        model_factory,
        curiosity_factory,
        normalize_state,
        normalize_reward,
        reporter,
    ):
        self.env = env
        self.reporter = reporter
        self.state_converter = Converter.for_space(self.env.observation_space)
        self.action_converter = Converter.for_space(self.env.action_space)
        self.model = model_factory.create(self.state_converter, self.action_converter)
        self.curiosity = curiosity_factory.create(
            self.state_converter, self.action_converter
        )
        self.reward_normalizer = (
            StandardNormalizer() if normalize_reward else NoNormalizer()
        )
        self.state_normalizer = (
            self.state_converter.state_normalizer()
            if normalize_state
            else NoNormalizer()
        )
        self.normalize_state = normalize_state
        self.device = None
        self.dtype = None
        self.numpy_dtype = None

    def act(self, state):
        state = self.state_normalizer.transform(state[:, None, :])
        reshaped_states = self.state_converter.reshape_as_input(state)
        logits = self.model.policy_logits(
            torch.tensor(reshaped_states, device=self.device)
        )
        return self.action_converter.action(logits).cpu().detach().numpy()

    def _train(
        self,
        states,
        actions,
        rewards,
        dones,
    ):
        raise NotImplementedError("Implement me")

    def learn(
        self,
        epochs,
        n_steps,
        initialization_steps=1000,
        render=False,
    ):
        if initialization_steps and self.normalize_state:
            s, _, _, _ = RandomRunner(self.env).run(initialization_steps)
            self.state_normalizer.partial_fit(s)

        for epoch in range(epochs):
            states, actions, rewards, dones = Runner(self.env, self).run(
                n_steps, render
            )
            states = self.state_normalizer.partial_fit_transform(states)
            rewards = self.curiosity.reward(rewards, states, actions)
            rewards = self.reward_normalizer.partial_fit_transform(rewards)
            self._train(states, actions, rewards, dones)
            print(f"Epoch: {epoch} done")

    def eval(self, n_steps, render=False):
        Runner(self.env, self).run(n_steps, render)

    def to(self, device, dtype, numpy_dtype):
        self.device = device
        self.dtype = dtype
        self.numpy_dtype = numpy_dtype
        self.model.to(device, dtype)
        self.curiosity.to(device, dtype)
        self.env.astype(numpy_dtype)

    def _tensors_to_device(self, *tensors):
        return [tensor.to(self.device, self.dtype) for tensor in tensors]

    def _to_tensor(self, array):
        return torch.tensor(array, device=self.device, dtype=self.dtype)
