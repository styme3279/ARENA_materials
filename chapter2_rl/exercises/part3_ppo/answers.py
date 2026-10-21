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
