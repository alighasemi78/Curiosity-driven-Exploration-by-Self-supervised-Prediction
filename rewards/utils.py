import numpy as np
from numba import jit


@jit(nopython=True, nogil=True)
def discount(
    rewards,
    estimate_of_last,
    dones,
    discount,
):
    v = estimate_of_last
    ret = np.zeros_like(rewards)
    for timestep in range(rewards.shape[1] - 1, -1, -1):
        r, done = rewards[:, timestep], dones[:, timestep]
        v = (r + discount * v * (1.0 - done)).astype(ret.dtype)
        ret[:, timestep] = v
    return ret
