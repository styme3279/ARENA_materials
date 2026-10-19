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
    %pip install "wandb>=0.18" einops "gymnasium[atari,other]>=1.2,<2" "ale-py>=0.10" pygame jaxtyping

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
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import einops
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tqdm.auto import tqdm

Arr: TypeAlias = np.ndarray

max_episode_steps = 1000
N_RUNS = 200

# Make sure exercises are in the path
chapter = "chapter2_rl"
section = "part1_intro_to_rl"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

import part1_intro_to_rl.tests as tests
import part1_intro_to_rl.utils as utils
from plotly_utils import cliffwalk_imshow, imshow, line
from part1_intro_to_rl.utils import GridWorld, Environment
# %%
def policy_eval_numerical(env: Environment, pi: Arr, gamma=0.99, eps=1e-8, max_iterations=10_000) -> Arr:
    """
    Numerically evaluates the value of a given policy by iterating the Bellman equation
    Args:
        env: Environment
        pi : shape (num_states,) - The policy to evaluate
        gamma: float - Discount factor
        eps  : float - Tolerance
        max_iterations: int - Maximum number of iterations to run
    Outputs:
        value : float (num_states,) - The value function for policy pi
    """
    num_states = env.num_states
    num_actions = env.num_actions

    # print(f"Evaluating policy for {num_states} states and {num_actions} actions")

    # for _ in range(max_iterations):
    transition_matrix = np.zeros((num_states, num_actions, num_states))
    reward_matrix = np.zeros((num_states, num_actions, num_states))
    for state in range(num_states):
        for action in range(num_actions):
            next_states, rewards, probs = env.dynamics(state, action)
            # print(f"statesS: {states}, rewards: {rewards}, probs: {probs}")
            for next_state, reward, prob in zip(next_states, rewards, probs):
                transition_matrix[state, action, next_state] += prob
                reward_matrix[state, action, next_state] = reward

    v_old = np.zeros(num_states)
    v_new = v_old.copy()
    for _ in range(max_iterations):
        for state in range(num_states):
            action = pi[state]
            next_states, _, _ = env.dynamics(state, action)
            state_value = 0
            for next_state in next_states:
                state_value += transition_matrix[state, action, next_state] * (reward_matrix[state, action, next_state] + gamma * v_old[next_state])
            v_new[state] = state_value

        if np.max(np.abs(v_new - v_old)) < eps:
            break
        v_old = v_new.copy()
    return v_old

            


tests.test_policy_eval(policy_eval_numerical, exact=False)
# %%
