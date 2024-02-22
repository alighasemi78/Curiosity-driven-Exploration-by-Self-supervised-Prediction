import numpy as np


class Runner:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self, n_steps, render=False):
        state = self.env.reset()
        states = np.empty(
            self.get_mini_batch_shape(state.shape, n_steps + 1), dtype=self.env.dtype
        )  # +1 for initial state
        rewards = np.empty(
            self.get_mini_batch_shape((self.env.n_envs,), n_steps), dtype=self.env.dtype
        )
        dones = np.empty(
            self.get_mini_batch_shape((self.env.n_envs,), n_steps), dtype=self.env.dtype
        )
        actions = None
        states[:, 0] = state
        for step in range(n_steps):
            if render:
                self.env.render()
            action = self.agent.act(state)
            if step == 0:  # lazy init when we know the action space shape
                actions = np.empty(
                    self.get_mini_batch_shape(action.shape, n_steps),
                    dtype=self.env.dtype,
                )
            state, reward, done, _ = self.env.step(action)
            states[:, step + 1] = state
            actions[:, step] = action
            rewards[:, step] = reward
            dones[:, step] = done
        return states, actions, rewards, dones

    def get_mini_batch_shape(self, observation_shape, n_steps):
        return (self.env.n_envs, n_steps, *observation_shape[1:])


class RandomRunner(Runner):
    def __init__(self, env):
        from agents import RandomAgent

        super().__init__(env, RandomAgent(env.action_space))
