from multiprocessing import Process, Pipe

import numpy as np
from gym import Env, make

from envs.utils import tile_images


class SubProcessEnv(Process):
    DONE_IDX = 2

    def __init__(self, env_id, master, slave):
        super().__init__(daemon=True)
        self.master = master
        self.env_id = env_id
        self.pipe = slave

    def start(self):
        super().start()
        self.pipe.close()

    def run(self):
        self.master.close()
        env = make(self.env_id, render_mode="rgb_array")
        steps = 0
        collected_reward = 0
        while True:
            command, args = self.pipe.recv()
            if command == "getattr":
                self.pipe.send(getattr(env, args))
            elif command == "seed":
                self.pipe.send(env.seed(args))
            elif command == "reset":
                steps = 0
                collected_reward = 0
                self.pipe.send(env.reset()[0])
            elif command == "render":
                self.pipe.send(env.render())
            elif command == "step":
                state, reward, terminated, truncated, aux = env.step(args)
                done = terminated or truncated
                steps += 1
                collected_reward += reward
                if done:
                    state, aux = env.reset()
                self.pipe.send((state, reward, done, aux, collected_reward))
                if done:
                    steps = 0
                    collected_reward = 0
            elif command == "close":
                env.close()
                break


class MultiEnv(Env):
    def __init__(self, env_id, n_envs, reporter):
        self._closed = False
        self.env_id = env_id
        self.n_envs = n_envs
        self.reporter = reporter
        self.processes = [SubProcessEnv(env_id, *Pipe()) for _ in range(self.n_envs)]
        self._start()
        self.observation_space = self._get_property(
            self.processes[0], "observation_space"
        )
        self.action_space = self._get_property(self.processes[0], "action_space")
        self.dtype = None

    def _start(self):
        for process in self.processes:
            process.start()

    def _send_command(self, name, args=None, await_response=True):
        for process, arg in zip(
            self.processes, args if args is not None else [None] * len(self.processes)
        ):
            process.master.send((name, arg))
        return (
            [self._rcv(process) for process in self.processes] if await_response else []
        )

    def _rcv(self, process):
        res = process.master.recv()
        return res

    def _get_property(self, process, name):
        process.master.send(("getattr", name))
        return self._rcv(process)

    def step(self, action):
        if len(action) != self.n_envs:
            raise ValueError("Not enough actions supplied")
        state, reward, done, aux, collected_rewards = zip(
            *self._send_command("step", action)
        )
        self._report_steps(done, collected_rewards)
        return (
            np.array(state, dtype=self.dtype),
            np.array(reward, dtype=self.dtype),
            np.array(done, dtype=self.dtype),
            aux,
        )

    def reset(self):
        return np.array(self._send_command("reset"), dtype=self.dtype)

    def render(self, mode="human"):
        imgs = self._send_command("render")
        if any(img is None for img in imgs):
            return None
        bigimg = tile_images(imgs)
        if mode == "human":
            import cv2

            cv2.imshow(self.env_id, bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def close(self):
        if self._closed:
            return
        self._send_command("close", await_response=False)
        for process in self.processes:
            process.join()
        self._closed = True

    def seed(self, seed=None):
        return np.array(
            self._send_command("seed", [seed] * self.n_envs), dtype=self.dtype
        )

    def astype(self, dtype):
        self.dtype = dtype

    def _report_steps(self, dones, collected_rewards):
        for done, reward in zip(dones, collected_rewards):
            if done:
                self.reporter.scalar("env/reward", reward)
