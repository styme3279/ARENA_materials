# %%
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # don't let JAX pre-grab all GPU memory
import einops
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import wandb
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from matplotlib.animation import FuncAnimation
from numpy.random import Generator
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.optim.optimizer import Optimizer
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Make sure exercises are in the path
chapter = "chapter2_rl"
section = "part3_ppo"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part3_ppo.tests as tests
from part1_intro_to_rl.utils import set_global_seeds
from part3_ppo.utils import arg_help
from plotly_utils import plot_cartpole_obs_and_dones
from rl_utils import AtariEnvs, BraxEnvs, LiveVideo, log_grid_video, log_video_html, record_grid_video, record_brax_video
from gpu_env import GPUEnv, CartPole, CartDoublePendulum, MountainCar, Pendulum, GPUProbe, angle_normalize, get_episode_data_from_infos

Arr = np.ndarray

# The networks in this section are tiny, and if you train on the CPU with
# PyTorch's default of one thread per core, threads spend longer
# coordinating than computing. Adjust if needed.
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
ENV_DICT = {"atari": AtariEnvs, "mujoco": BraxEnvs, "classic-control": CartPole, "swing-up": CartDoublePendulum,
            "mountain-car": MountainCar, "pendulum": Pendulum, "probe": GPUProbe}
EnvType = Literal["atari", "mujoco", "classic-control", "swing-up", "mountain-car", "pendulum", "probe"]
BatchedEnvs = GPUEnv | AtariEnvs | BraxEnvs  # anything `ENV_DICT` can build; all batched & tensor-native
# The bonus training loops below (Atari / MuJoCo / swing-up) each take a few minutes. They're guarded
# by `if SLOW:` so a top-to-bottom run of this file trains only the fast (~15s) CartPole; flip this to
# True to actually run the bonus environments.
SLOW = False

# %%
@dataclass
class PPOArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "CartPole-v1"
    mode: EnvType = "classic-control"
    device: str = device  # envs, networks and replay memory all live here; "cpu" is fastest for CartPole

    # Wandb / logging
    use_wandb: bool = False
    video_log_freq: int | None = 50  # every N phases, render a 4x4 grid video of the latest rollout
                                     # (logged to wandb if use_wandb, shown inline if live_viz)
    live_viz: bool = True  # display the 4x4 grid video inline every `video_log_freq` phases (notebook only)
    overwrite_video: bool = True  # each live video replaces the previous one in the notebook output (False appends them)
    wandb_project_name: str = "PPOCartPole"
    wandb_entity: str | None = None

    # Duration of training. One phase = a rollout of num_envs * num_steps_per_rollout env steps, then
    # num_minibatches * batches_per_learning_phase gradient steps on it. CartPole is solved after ~20 phases
    # (~2 s on CPU, ~4 s on a GPU); more parallel envs make each phase's batch bigger but don't help here.
    total_phases: int = 100
    num_envs: int = 64
    num_steps_per_rollout: int = 64
    num_minibatches: int = 4
    batches_per_learning_phase: int = 4

    # Optimization hyperparameters (a high LR converges in seconds on CartPole)
    lr: float = 5e-3
    max_grad_norm: float = 0.5

    # RL hyperparameters
    gamma: float = 0.99

    # PPO-specific hyperparameters
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 1.0

    def __post_init__(self):
        self.batch_size = self.num_steps_per_rollout * self.num_envs

        assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.total_timesteps = self.total_phases * self.batch_size
        self.total_training_steps = self.total_phases * self.batches_per_learning_phase * self.num_minibatches
        self.device = t.device(self.device)


args = PPOArgs(num_minibatches=2)  # changing this also changes minibatch_size and total_training_steps
arg_help(args)

# %%
def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_actor_and_critic(
    envs: BatchedEnvs,
    mode: EnvType = "classic-control",
) -> tuple[nn.Module, nn.Module]:
    """
    Returns (actor, critic), the networks used for PPO, using one of 3 different architectures depending on `mode`.
    """
    assert mode in ENV_DICT

    # Observations are continuous: `obs_shape` is the shape of one observation (e.g. (4,) for CartPole,
    # (4, 84, 84) for Atari), and `obs_dim` is the number of floats in it once flattened (4 for CartPole).
    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_space = envs.single_action_space

    if mode in ("classic-control", "mountain-car", "probe"):
        # mountain-car (Discrete(3)) and the probe envs are also discrete classic-control tasks, so
        # they reuse this network. Discrete actions: `num_actions = action_space.n` is how many there are.
        num_actions = action_space.n
        actor, critic = get_actor_and_critic_classic(obs_dim, num_actions)
    elif mode == "atari":
        num_actions = action_space.n
        actor, critic = get_actor_and_critic_atari(obs_shape, num_actions)  # you'll implement these later
    elif mode in ("mujoco", "swing-up", "pendulum"):
        # swing-up (cart + double-pendulum) and pendulum are continuous-action tasks, so they reuse
        # the MuJoCo Gaussian actor/critic. Continuous actions: `action_dim` is the length of the action vector.
        action_dim = int(np.prod(action_space.shape))
        actor, critic = get_actor_and_critic_mujoco(obs_dim, action_dim)  # you'll implement these later
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    return actor.to(device), critic.to(device)


def get_actor_and_critic_classic(obs_dim: int, num_actions: int):
    """
    Returns (actor, critic) in the "classic-control" case, according to diagram above.
    """

    actor = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01)
    )

    critic = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1.0)
    )

    return (actor, critic)


tests.test_get_actor_and_critic(get_actor_and_critic, mode="classic-control")

# %%
@t.inference_mode()
def compute_advantages(
    next_value: Float[Tensor, "num_envs"],
    next_terminated: Bool[Tensor, "num_envs"],
    rewards: Float[Tensor, "buffer_size num_envs"],
    values: Float[Tensor, "buffer_size num_envs"],
    terminated: Bool[Tensor, "buffer_size num_envs"],
    gamma: float,
    gae_lambda: float,
) -> Float[Tensor, "buffer_size num_envs"]:
    """
    Compute advantages using Generalized Advantage Estimation.
    """
    buffer_size = rewards.shape[0]
    num_envs = rewards.shape[1]

    advantages = t.zeros((buffer_size, num_envs)).to(rewards.device)

    current_sum = t.zeros(num_envs).to(rewards.device)

    next_terminated = next_terminated.float()


    for time in reversed(range(buffer_size)):

        td_error = rewards[time] + (1 - next_terminated) * gamma * next_value - values[time]

        current_sum = td_error + (1 - next_terminated) * gamma * gae_lambda * current_sum

        advantages[time] = current_sum

        next_terminated = terminated[time].float()
        next_value = values[time]

    return advantages

tests.test_compute_advantages(compute_advantages)

# %%
def get_minibatch_indices(rng: Generator, batch_size: int, minibatch_size: int) -> list[np.ndarray]:
    """
    Return a list of length `num_minibatches`, where each element is an array of `minibatch_size` and the union of all
    the arrays is the set of indices [0, 1, ..., batch_size - 1] where `batch_size = num_steps_per_rollout * num_envs`.
    """
    assert batch_size % minibatch_size == 0
    indices = rng.permutation(batch_size)
    num_batches = batch_size / minibatch_size
    return np.split(indices, num_batches)

rng = np.random.default_rng(0)

batch_size = 12
minibatch_size = 6
# num_minibatches = batch_size // minibatch_size = 2

indices = get_minibatch_indices(rng, batch_size, minibatch_size)

assert isinstance(indices, list)
assert all(isinstance(x, np.ndarray) for x in indices)
assert np.array(indices).shape == (2, 6)
assert sorted(np.unique(indices)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print("All tests in `test_minibatch_indexes` passed!")

# %%
@dataclass
class ReplayMinibatch:
    """
    Samples from the replay memory, converted to PyTorch for use in neural network training.

    Data is equivalent to (s_t, a_t, logpi(a_t|s_t), A_t, A_t + V(s_t), d_t)
    """

    obs: Float[Tensor, " minibatch_size *obs_shape"]
    actions: Int[Tensor, " minibatch_size *action_shape"]
    logprobs: Float[Tensor, " minibatch_size"]
    advantages: Float[Tensor, " minibatch_size"]
    returns: Float[Tensor, " minibatch_size"]
    terminated: Bool[Tensor, " minibatch_size"]


class ReplayMemory:
    """
    Contains buffer; has a method to sample from it to return a ReplayMinibatch object.
    """

    rng: Generator
    obs: Float[Arr, " buffer_size num_envs *obs_shape"]
    actions: Int[Arr, " buffer_size num_envs *action_shape"]
    logprobs: Float[Arr, " buffer_size num_envs"]
    values: Float[Arr, " buffer_size num_envs"]
    rewards: Float[Arr, " buffer_size num_envs"]
    terminated: Bool[Arr, " buffer_size num_envs"]

    def __init__(
        self,
        num_envs: int,
        obs_shape: tuple,
        action_shape: tuple,
        batch_size: int,
        minibatch_size: int,
        batches_per_learning_phase: int,
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.batches_per_learning_phase = batches_per_learning_phase
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Resets all stored experiences, ready for new ones to be added to memory. We keep one
        list of per-step GPU tensors per field and stack them once at sampling time, so nothing
        leaves the GPU during a rollout."""
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.terminated = []

    def add(
        self,
        obs: Float[Arr, " num_envs *obs_shape"],
        actions: Int[Arr, " num_envs *action_shape"],
        logprobs: Float[Arr, " num_envs"],
        values: Float[Arr, " num_envs"],
        rewards: Float[Arr, " num_envs"],
        terminated: Bool[Arr, " num_envs"],
    ) -> None:
        """Add a batch of transitions to the replay memory. All inputs are GPU tensors and stay on
        the GPU (we just append the references; they're stacked at sampling time)."""
        # Check shapes & datatypes
        for data, expected_shape in zip(
            [obs, actions, logprobs, values, rewards, terminated],
            [self.obs_shape, self.action_shape, (), (), (), ()],
        ):
            assert isinstance(data, Tensor)
            assert data.shape == (self.num_envs, *expected_shape)

        # Append this step's tensors (no concatenation / host transfer)
        self.obs.append(obs)
        self.actions.append(actions)
        self.logprobs.append(logprobs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.terminated.append(terminated)

    def get_minibatches(
        self, next_value: Tensor, next_terminated: Tensor, gamma: float, gae_lambda: float
    ) -> list[ReplayMinibatch]:
        """
        Returns a list of minibatches. Each minibatch has size `minibatch_size`, and the union over
        all minibatches is `batches_per_learning_phase` copies of the entire replay memory.

        Note this does more than just sampling: it first computes the advantages (GAE) and returns from
        the stored rollout, and it already repeats the shuffle-and-split `batches_per_learning_phase`
        times. So you don't need to compute advantages or loop over learning epochs yourself.
        """
        # Stack the per-step tensors into shape (buffer_size, num_envs, ...)
        obs, actions, logprobs, values, rewards, terminated = (
            t.stack(x)
            for x in [
                self.obs,
                self.actions,
                self.logprobs,
                self.values,
                self.rewards,
                self.terminated,
            ]
        )

        # Compute advantages & returns
        advantages = compute_advantages(next_value, next_terminated, rewards, values, terminated, gamma, gae_lambda)
        returns = advantages + values

        # Return a list of minibatches (indices are moved onto the data's device to index it)
        minibatches = []
        for _ in range(self.batches_per_learning_phase):
            for indices in get_minibatch_indices(self.rng, self.batch_size, self.minibatch_size):
                idx = t.as_tensor(indices, dtype=t.long, device=obs.device)
                minibatches.append(
                    ReplayMinibatch(
                        *[
                            data.flatten(0, 1)[idx]
                            for data in [obs, actions, logprobs, advantages, returns, terminated]
                        ]
                    )
                )

        # Reset memory (since we only need to call this method once per learning phase)
        self.reset()

        return minibatches

# %%
num_steps_per_rollout = 128
num_envs = 2
batch_size = num_steps_per_rollout * num_envs  # 256

minibatch_size = 128
num_minibatches = batch_size // minibatch_size  # 2

batches_per_learning_phase = 2

envs = CartPole(num_envs=num_envs)  # GPU-batched env: reset()/step() return on-device tensors
memory = ReplayMemory(num_envs, (4,), (), batch_size, minibatch_size, batches_per_learning_phase)

logprobs = values = t.zeros(num_envs, device=device)  # dummy values, just so we can see demo of plot
obs, _ = envs.reset()
terminated = t.zeros(num_envs, dtype=t.bool, device=device)  # d_t for the very first obs

for i in range(num_steps_per_rollout):  # the locally-defined 128, matching this cell's batch_size
    # Choose a random action and step the env (actions are an on-device tensor for the batched env)
    actions = t.randint(0, envs.single_action_space.n, (num_envs,), device=device)
    next_obs, rewards, next_terminated, truncated, infos = envs.step(actions)

    # Add experience to memory (everything is already an on-device tensor). Like `PPOAgent.play_step`,
    # the flag we store next to s_t is d_t, i.e. "s_t is the first obs of a new episode".
    memory.add(obs, actions, logprobs, values, rewards, terminated)
    obs, terminated = next_obs, next_terminated

plot_cartpole_obs_and_dones(
    t.stack(memory.obs).cpu(),
    t.stack(memory.terminated).cpu(),
    title="Current obs s<sub>t</sub><br>Dotted lines indicate d<sub>t</sub> = 1, solid lines are environment separators",
)

next_value = next_done = t.zeros(envs.num_envs).to(device)  # dummy values, just so we can see demo of plot
minibatches = memory.get_minibatches(next_value, next_done, gamma=0.99, gae_lambda=0.95)

plot_cartpole_obs_and_dones(
    minibatches[0].obs.cpu(),
    minibatches[0].terminated.cpu(),
    title="Current obs (sampled)<br>this is what gets fed into our model for training",
)

# %%
class PPOAgent:
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(
        self,
        envs: BatchedEnvs,
        actor: nn.Module,
        critic: nn.Module,
        memory: ReplayMemory,
    ):
        super().__init__()
        self.envs = envs
        self.actor = actor
        self.critic = critic
        self.memory = memory

        self.step = 0  # Tracking number of steps taken (across all environments)
        self.next_obs = t.as_tensor(envs.reset()[0], dtype=t.float)  # need starting obs (already on the env's device)
        self.next_terminated = t.zeros(envs.num_envs, device=self.next_obs.device, dtype=t.bool)  # need starting termination=False

    def play_step(self) -> list[dict]:
        """
        Carries out a single interaction step between the agent and the environment, and adds
        results to the replay memory.

        Returns the list of info dicts returned from `self.envs.step`.
        """
        # Get newest observations (i.e. where we're starting from)
        obs = self.next_obs
        terminated = self.next_terminated

        with t.inference_mode():
            action_logits = self.actor(obs) # (num_envs, action_space)
            values = self.critic(obs).squeeze(-1) # (num_envs, 1)

        dist = Categorical(logits=action_logits)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)


        self.next_obs, rewards, next_terminated, next_truncated, infos =  self.envs.step(actions)

        self.next_terminated = next_terminated #| next_truncated


        self.memory.add(
            obs, actions, logprobs, values, rewards, terminated
        )
        
        self.step += self.envs.num_envs
        return infos

    def get_minibatches(self, gamma: float, gae_lambda: float) -> list[ReplayMinibatch]:
        """
        Gets minibatches from the replay memory, and resets the memory
        """
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        minibatches = self.memory.get_minibatches(next_value, self.next_terminated, gamma, gae_lambda)
        self.memory.reset()
        return minibatches


tests.test_ppo_agent(PPOAgent)

# %%
def calc_clipped_surrogate_objective(
    dist: Categorical,
    mb_action: Int[Tensor, "minibatch_size"],
    mb_advantages: Float[Tensor, "minibatch_size"],
    mb_logprobs: Float[Tensor, "minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    """Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    dist:
        a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
    mb_action:
        what actions were taken in the sampled minibatch
    mb_advantages:
        advantages calculated from the sampled minibatch
    mb_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    """
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape

    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

    new_logprobs = dist.log_prob(mb_action) # (minibatch_size)
    iw = np.exp(new_logprobs - mb_logprobs) # (minibatch_size)

    weighted_advantages = iw * mb_advantages

    clipped_iw = np.clip(iw, 1 - clip_coef, 1 + clip_coef)
    clipped_weighted_advantages = clipped_iw * mb_advantages

    min_advantage = np.minimum(weighted_advantages, clipped_weighted_advantages) # (minibatch_size)

    loss = min_advantage.mean() 

    return loss

tests.test_calc_clipped_surrogate_objective(calc_clipped_surrogate_objective)


# %%
