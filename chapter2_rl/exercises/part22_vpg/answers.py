# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter2_rl"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
try:
    import jaxtyping
except:
    %pip install "wandb>=0.18" einops "gymnasium[atari,other]>=1.2,<2" "ale-py>=0.10" pygame jaxtyping torchinfo git+https://github.com/ARENA-education/eindex.git

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
    if not IN_COLAB:
        !sudo apt-get install unzip
        %pip install jupyter ipython --upgrade

    if not os.path.exists(f"{root}/{chapter}"):
        !wget -P {root} https://github.com/ARENA-education/ARENA_materials/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/{repo}-{branch}


if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")

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
        self.layers = t.nn.Sequential(
            t.nn.Linear(obs_shape[0], hidden_sizes[0]),
            t.nn.ReLU(),
            t.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            t.nn.ReLU(),
            t.nn.Linear(hidden_sizes[1], num_actions),
        )

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

        for buffer, info in [(self._obs, obs), (self._actions, actions), (self._logprobs, logprobs), (self._rewards, rewards), (self._dones, dones)]:
            buffer.append(info)
        
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
        tau = self.get()  # filled portion only
        obs = t.split(tau.obs, batch_size, dim=0)
        acts = t.split(tau.actions, batch_size, dim=0)
        logprobs = t.split(tau.logprobs, batch_size, dim=0)
        rewards = t.split(tau.rewards, batch_size, dim=0)
        dones = t.split(tau.dones, batch_size, dim=0)

        batches = [RolloutTensors(*tensors) for tensors in zip(obs, acts, logprobs, rewards, dones)]

        return batches



tests.test_rollout(Rollout)
# %%
