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
class Norvig(GridWorld):
    """The 3x4 Russell & Norvig gridworld, expressed as a `GridWorld` map: a +1 goal in the top-right
    with a -1 trap directly below it, a wall in the middle, a small step penalty everywhere else, and
    `slipperiness=0.3` (the chosen action succeeds 70% of the time)."""

    def __init__(self, penalty=-0.04):
        super().__init__(
            """
            ...G
            .#.T
            S...
            """,
            step_reward=penalty,
            goal_reward=1.0,
            trap_reward=-1.0,
            slipperiness=0.3,
        )


norvig = Norvig()
tests.test_norvig_gridworld_equivalence(Norvig)

# `render` prints the grid; `show_map` plots it. Here we show a random policy:
pi_random = np.random.randint(0, norvig.num_actions, (norvig.num_states,))
norvig.show_map(pi_random, title="Norvig gridworld: random policy")
# %%
# def policy_eval_numerical(env: Environment, pi: Arr, gamma=0.99, eps=1e-8, max_iterations=10_000) -> Arr:
# """
# Numerically evaluates the value of a given policy by iterating the Bellman equation
# Args:
#     env: Environment
#     pi : shape (num_states,) - The policy to evaluate
#     gamma: float - Discount factor
#     eps  : float - Tolerance
#     max_iterations: int - Maximum number of iterations to run
# Outputs:
#     value : float (num_states,) - The value function for policy pi
# """
# num_states = env.num_states
# num_actions = env.num_actions

# # print(f"Evaluating policy for {num_states} states and {num_actions} actions")

# # for _ in range(max_iterations):
# transition_matrix = np.zeros((num_states, num_actions, num_states))
# reward_matrix = np.zeros((num_states, num_actions, num_states))

# # print(env.T.shape, env.R.shape)

# for state in range(num_states):
#     for action in range(num_actions):
#         next_states, rewards, probs = env.dynamics(state, action)
#         # print(f"statesS: {states}, rewards: {rewards}, probs: {probs}")
#         for next_state, reward, prob in zip(next_states, rewards, probs):
#             transition_matrix[state, action, next_state] += prob
#             reward_matrix[state, action, next_state] += reward

# if not np.allclose(env.T, transition_matrix):
#     print("Transition matrix does not match")
#     print("Expected:")
#     print(env.T)
#     print("Got:")
#     print(transition_matrix)
# assert np.allclose(env.T, transition_matrix)
# if not np.allclose(env.R, reward_matrix):
#     print("Reward matrix does not match")
#     print("Expected:")
#     print(env.R)
#     print("Got:")
#     print(reward_matrix)
# assert np.allclose(env.R, reward_matrix)

# v_old = np.zeros(num_states)
# v_new = v_old.copy()
# for _ in range(max_iterations):
#     for state in range(num_states):
#         action = pi[state]
#         state_value = 0
#         for next_state in range(num_states):
#             state_value += transition_matrix[state, action, next_state] * (reward_matrix[state, action, next_state] + gamma * v_old[next_state])
#         v_new[state] = state_value

#     if np.max(np.abs(v_new - v_old)) < eps:
#         break
#     v_old = v_new.copy()
# return v_old

def policy_eval_numerical(env: Environment, pi: Arr, gamma=0.99, eps=1e-8,max_iterations=10_000) -> Arr:
    states = np.arange(env.num_states)
    P = env.T[states, pi]  # (S, S): P[s, s'] = T(s' | s, pi(s))
    R = env.R[states, pi]  # (S, S): R[s, s'] = R(s, pi(s), s')

    V = np.zeros(env.num_states)
    for _ in range(max_iterations):
        V_new = (P * (R + gamma * V)).sum(axis=1)
        if np.abs(V_new - V).max() < eps:
            return V_new
        V = V_new
    return V

tests.test_policy_eval(policy_eval_numerical, exact=False)

# %%
def policy_eval_exact(env: Environment, pi: Arr, gamma=0.99) -> Arr:
    """
    Finds the exact solution to the Bellman equation.
    """
    num_states = env.num_states

    transition_matrix = env.T[range(num_states), pi, :]  # shape [s, s_next]
    reward_matrix = env.R[range(num_states), pi, :]  # shape [s, s_next]

    r = einops.einsum(transition_matrix, reward_matrix, "s s_next, s s_next -> s")

    mat = np.eye(num_states) - gamma * transition_matrix

    return np.linalg.solve(mat, r)


tests.test_policy_eval(policy_eval_exact, exact=True)
# %%
def policy_improvement(env: Environment, V: Arr, gamma=0.99) -> Arr:
    """
    Args:
        env: Environment
        V  : (num_states,) value of each state following some policy pi
    Outputs:
        pi_better : vector (num_states,) of actions representing a new policy obtained via policy
                    iteration
    """
    pi_better = np.zeros(env.num_states, dtype=int)
    for state in range(env.num_states):
        best_action = 0
        best_q_value = -np.inf
        for action in range(env.num_actions):
            next_states, rewards, probs = env.dynamics(state, action)
            action_value = sum(prob * (reward + gamma * V[next_state]) for next_state, reward, prob in zip(next_states, rewards, probs))
            # action_value = sum(env.T[state, action, next_state] * (env.R[state, action, next_state] + gamma * V[next_state]) for next_state in range(env.num_states))
            if action_value > best_q_value:
                best_q_value = action_value
                best_action = action
        pi_better[state] = best_action
    return pi_better
tests.test_policy_improvement(policy_improvement)
# %%
def find_optimal_policy(env: Environment, gamma=0.99, max_iterations=10_000):
    """
    Args:
        env: environment
    Outputs:
        pi : (num_states,) int, of actions representing an optimal policy
    """
    pi = np.zeros(shape=env.num_states, dtype=int)
    for _ in range(max_iterations):
        V = policy_eval_exact(env, pi, gamma)
        pi_new = policy_improvement(env, V, gamma)
        if np.array_equal(pi_new, pi):
            break
        pi = pi_new
    return pi


tests.test_find_optimal_policy(find_optimal_policy)

penalty = -0.04
norvig = Norvig(penalty)
pi_opt = find_optimal_policy(norvig, gamma=0.99)
norvig.show_map(pi_opt, title="Optimal Policy, penalty = -0.04")
# %%
