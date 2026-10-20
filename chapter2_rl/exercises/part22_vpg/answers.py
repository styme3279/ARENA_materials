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
@dataclass
class VPGArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "CartPole-gpu"

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "VPGCartPole"
    wandb_entity: str | None = None
    video_log_freq: int | None = 50   # every N gradient steps, render a 4x4 grid video of the latest rollout
                                      # (logged to wandb if use_wandb, shown inline if live_viz)

    # Duration of different phases / buffer memory settings
    total_timesteps: int = 500_000
    # max_rollout_steps: int = 500
    # min_rollout_steps: int = 64
    num_envs: int = 4

    num_steps_per_rollout: int = 128

    lr: float = 2.5e-4
    gamma: float = 1
    rollout_use_count: int = 1
    clip_coef: float = 0.2
    device: str = "cpu"
    normalize_returns: bool = True
    num_batches_per_rollout: int = 1
    # LR decay settings
    use_lr_decay: bool = False
    lr_end: Optional[float] = None
    lr_frac: Optional[float] = None
    use_iw: bool = False
    full_reset: bool = True   # fully reset all envs at the start of each rollout
    live_viz: bool = True     # display the 4x4 grid video inline every `video_log_freq` gradient steps (notebook only)
    overwrite_video: bool = True  # each live video replaces the previous one in the notebook output (False appends them)

    def __post_init__(self):
        self.batch_size = self.num_envs // self.num_batches_per_rollout
        self.device = t.device(self.device)

        if self.use_lr_decay:
            assert self.lr_end is not None, "lr_end must be set if use_lr_decay is True"
            assert self.lr_frac is not None, "lr_frac must be set if use_lr_decay is True"

        self.env_steps_per_update = self.num_steps_per_rollout * self.num_envs // self.num_batches_per_rollout

        if not self.use_iw:
            assert self.rollout_use_count == 1, "rollout_use_count must be 1 if use_iw is False"
            assert self.num_batches_per_rollout == 1, "num_batches_per_rollout must be 1 if use_iw is False"

# %%
class VPGAgent:
    """Base Agent class handling the interaction with the environment."""

    def __init__(
        self,
        envs: gym.Env,
        policy_network: PolicyNetwork,
        args: VPGArgs,
    ):
        self.envs = envs
        self.policy_network = policy_network
        self.args = args
        self.obs_shape = envs.observation_space.shape
        self.action_shape = envs.action_space.shape

    @t.no_grad()
    def gen_rollout(self, rollout: Rollout) -> Rollout:
        """
        Steps all environments in parallel for `args.num_steps_per_rollout` steps, adding each step to the rollout
        buffer (which is reset first), and returns the filled buffer.
        """
        if self.args.full_reset and hasattr(self.envs, "_terminated"):
            self.envs._terminated[:] = True
            self.envs._truncated[:] = True
        obs, _ = self.envs.reset()  # Need a starting observation
        rollout.reset()

        for _ in range(self.args.num_steps_per_rollout):
            actions, logprobs = self.get_actions(obs)
            
            obs, rewards, terminated, truncated, _ = self.envs.step(actions)

            rollout.add_step(obs, actions, logprobs, rewards, terminated | truncated)

        return rollout

    def get_actions(
        self, obs: Float[Tensor, " num_envs *obs_shape"]
    ) -> tuple[Int[Tensor, " num_envs"], Float[Tensor, " num_envs"]]:
        """
        Computes the agent's turn: given an observation for each environment,
        sample the action the agent takes, along with the log_probs of that action.
        Use t.multinomial to sample the actions.
        """
        qs = self.policy_network(obs)  # (256, 2)

        chosen = t.multinomial(qs.softmax(dim=-1), num_samples=1)  # (256, 1)

        logprobs = qs.log_softmax(dim=-1)  # (256, 2)

        selected_logprobs = logprobs.gather(dim=1, index=chosen)  # (256, 1)

        return chosen.squeeze(-1), selected_logprobs.squeeze(-1)


tests.test_get_actions(VPGAgent, PolicyNetwork)
tests.test_gen_rollout(VPGAgent, PolicyNetwork, VPGArgs, Rollout)
# %%
