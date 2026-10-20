# %%

import sys
import time
import warnings
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
import torch as t
import torch.nn.functional as F
import wandb
from eindex import eindex
from gymnasium.spaces import Box, Discrete
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torchinfo import summary
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# Make sure exercises are in the path
chapter = "chapter2_rl"
section = "part22_vpg"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from gpu_env import CartPole, MountainCar
from gpu_probe import Probe1, Probe2, Probe3, Probe4, Probe5
from rl_utils import ENVS, AtariEnvs, LiveVideo, log_greedy_rollout_video, log_grid_video, make_envs
import part22_vpg.tests as tests
from part1_intro_to_rl.utils import set_global_seeds
from plotly_utils import line, plot_cartpole_obs_and_dones

# The networks in this section are tiny, and if you train on the CPU with
# PyTorch's default of one thread per core, threads spend longer
# coordinating than computing. Adjust if needed.
t.set_num_threads(min(4, t.get_num_threads()))

# CartPole will train just fine on a CPU as we need few parallel environments, and the neural network is very small
# Feel free to change this to "cpu" if you find it runs faster
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


MAIN = __name__ == "__main__"

# %%
class PolicyNetwork(nn.Module):
    """
    For consistency with your tests, please wrap your modules in a `nn.Sequential` called `layers`.
    """

    layers: nn.Sequential

    def __init__(self, obs_shape: tuple[int], num_actions: int, hidden_sizes: list[int] = [120, 84]):
        super().__init__()
        assert len(obs_shape) == 1, "Expecting a single vector of observations"
        in_features_list = [obs_shape[0]] + hidden_sizes
        out_features_list = hidden_sizes + [num_actions]
        layers = []
        for i, (in_features, out_features) in enumerate(zip(in_features_list, out_features_list)):
            layers.append(nn.Linear(in_features, out_features))
            if i < len(in_features_list) - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


net = PolicyNetwork(obs_shape=(4,), num_actions=2)
n_params = sum((p.nelement() for p in net.parameters()))
assert isinstance(getattr(net, "layers", None), nn.Sequential)
print(net)
print(f"Total number of parameters: {n_params}")
print("You should manually verify network is Linear-ReLU-Linear-ReLU-Linear")
assert not isinstance(net.layers[-1], nn.ReLU)
assert n_params == 10934

# %%
RolloutTensors = namedtuple("RolloutTensors", ["obs", "actions", "logprobs", "rewards", "dones"])


class Rollout:
    _obs: list[Float[Tensor, " num_envs *obs_shape"]]
    _actions: list[Int[Tensor, " num_envs"]]
    _logprobs: list[Float[Tensor, " num_envs"]]
    _rewards: list[Float[Tensor, " num_envs"]]
    _dones: list[Bool[Tensor, " num_envs"]]
    timestep: int

    def __init__(self, max_steps: int):
        """
        Args:
            max_steps: maximum number of steps to rollout per environment (the per-step tensors carry the
                       number of environments and the observation/action shapes themselves)
        """

        self.MAX_SIZE = max_steps

        # Per-step we append tensor references to Python lists (free) and t.stack() once at the
        # end, instead of 5 indexed-write kernels per step into a preallocated buffer. Each stored
        # tensor is freshly produced per step (the env returns a new state tensor each step), so
        # holding references is safe. This removes ~2500 tiny kernel launches per full rollout.
        self._obs, self._actions, self._logprobs, self._rewards, self._dones = [], [], [], [], []
        self.timestep = 0

    def add_step(
        self,
        obs: Float[Tensor, " num_envs *obs_shape"],
        actions: Int[Tensor, " num_envs"],
        logprobs: Float[Tensor, " num_envs"],
        rewards: Float[Tensor, " num_envs"],
        dones: Bool[Tensor, " num_envs"],
    ):
        """
        Stores one timestep of the rollout (the same transition for all `num_envs` environments at once)
        and advances `self.timestep`.

        Args:
            obs:      (num_envs, *obs_shape)  the observation (o_t) the agent saw *before* acting
            actions:  (num_envs,)             the action (a_t) it took in each env after seeing o_t
            logprobs: (num_envs,)             log pi(a_t | o_t) of the action that was *taken* 
                                              Recorded at collection time, so
                                              these are the behaviour policy's logprobs (used for importance weights)
            rewards:  (num_envs,)             the reward (r_{t+1}) received for taking a_t
            dones:    (num_envs,)  bool       True if that env's episode ended on this step (d_{t+1} = terminated OR truncated),
                                              i.e. the env auto-reset and the *next* stored obs starts a new episode

        Raises ValueError if the rollout already holds `max_steps` timesteps.
        """

        if self.timestep >= self.MAX_SIZE:
            raise ValueError("Rollout is full, cannot add more steps")
        self._obs.append(obs)
        self._actions.append(actions)
        self._logprobs.append(logprobs)
        self._rewards.append(rewards)
        self._dones.append(dones)
        self.timestep += 1

    def reset(self):
        self._obs.clear(); self._actions.clear(); self._logprobs.clear()
        self._rewards.clear(); self._dones.clear()
        self.timestep = 0

    def get(self) -> tuple[Tensor, ...]:
        """
        Stack the per-step lists from (num_envs, *tensor_shape) into (num_envs, timestep, *tensor_shape) tensors. Rollouts can stop early
        (see gen_rollout), so the time dimension is however many steps were actually collected.
        """
        assert self.timestep > 0, "Rollout is empty"
        return RolloutTensors(
            t.stack(self._obs, dim=1),
            t.stack(self._actions, dim=1),
            t.stack(self._logprobs, dim=1),
            t.stack(self._rewards, dim=1).float(),
            t.stack(self._dones, dim=1),
        )

    def get_batches(self, batch_size: int) -> list[RolloutTensors]:
        """
        Splits the rollout buffer into batches of size `batch_size`, and returns a list of
        `RolloutTensors` objects, each containing `batch_size` many trajectories.
        """

        res = self.get()
        batches = []
        for i in range(0, res.obs.shape[0], batch_size):
            for att in ["obs", "actions", "logprobs", "rewards", "dones"]:
                res[att][i:i+1]
                RolloutTensors()

        print(res.obs.shape)
        return res


tests.test_rollout(Rollout)
