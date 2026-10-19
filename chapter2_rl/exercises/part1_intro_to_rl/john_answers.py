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
class Environment:
    def __init__(self, num_states: int, num_actions: int, start=0, terminal=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.start = start
        self.terminal = np.array([], dtype=int) if terminal is None else terminal
        (self.T, self.R) = self._build()
        # self.T : (num_states, num_actions, num_states) transition probabilities
        # self.R : (num_states, num_actions, num_states) reward function
        
    def render(pi: Arr):
        """
        Takes a policy pi, and draws an image of the behavior of that policy, if applicable.

        Args:
            pi : (num_actions,) a policy

        Returns:
            None
        """
        raise NotImplementedError()
        
    def dynamics(self, state: int, action: int) -> tuple[Arr, Arr, Arr]:
        """
        Computes the distribution over possible outcomes for a given state
        and action.

        Args:
            state  : int (index of state)
            action : int (index of action)

        Returns:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        """
        raise NotImplementedError()

class Toy(Environment):
    def dynamics(self, state: int, action: int):
        """
        Sets up dynamics for the toy environment:
            - In state s_L, we move to s_0 & get +0 reward regardless of action
            - In state s_R, we move to s_0 & get +2 reward regardless of action
            - In state s_0,
                - action LEFT=0 leads to s_L & get +1,
                - action RIGHT=1 leads to s_R & get +0
        """
        (SL, S0, SR) = (0, 1, 2)
        LEFT = 0

        assert 0 <= state < self.num_states and 0 <= action < self.num_actions

        if state == S0:
            (next_state, reward) = (SL, 1) if action == LEFT else (SR, 0)
        elif state == SL:
            (next_state, reward) = (S0, 0)
        elif state == SR:
            (next_state, reward) = (S0, 2)
        else:
            raise ValueError(f"Invalid state: {state}")

        return (np.array([next_state]), np.array([reward]), np.array([1]))

    def __init__(self):
        super().__init__(num_states=3, num_actions=2)
# %%
toy = Toy()

actions = ["a_L", "a_R"]
states = ["s_L", "s_0", "s_R"]

imshow(
    toy.T,  # dimensions (s, a, s_next)
    title="Transition probabilities T(s_next | s, a) for toy environment",
    facet_col=0,
    facet_labels=[f"Current state is s = {s}" for s in states],
    y=actions,
    x=states,
    labels={
        "x": "Next state, s_next",
        "y": "Action taken, a",
        "color": "Transition<br>Probability",
    },
    text_auto=".0f",
    border=True,
    width=850,
    height=350,
)

imshow(
    toy.R,  # dimensions (s, a, s_next)
    title="Rewards R(s, a, s_next) for toy environment",
    facet_col=0,
    facet_labels=[f"Current state is s = {s}" for s in states],
    y=actions,
    x=states,
    labels={"x": "Next state, s_next", "y": "Action taken, a", "color": "Reward"},
    text_auto=".0f",
    border=True,
    width=850,
    height=350,
)
# %%

toy = Toy()

actions = ["a_L", "a_R"]
states = ["s_L", "s_0", "s_R"]

imshow(
    toy.T,  # dimensions (s, a, s_next)
    title="Transition probabilities T(s_next | s, a) for toy environment",
    facet_col=0,
    facet_labels=[f"Current state is s = {s}" for s in states],
    y=actions,
    x=states,
    labels={
        "x": "Next state, s_next",
        "y": "Action taken, a",
        "color": "Transition<br>Probability",
    },
    text_auto=".0f",
    border=True,
    width=850,
    height=350,
)

imshow(
    toy.R,  # dimensions (s, a, s_next)
    title="Rewards R(s, a, s_next) for toy environment",
    facet_col=0,
    facet_labels=[f"Current state is s = {s}" for s in states],
    y=actions,
    x=states,
    labels={"x": "Next state, s_next", "y": "Action taken, a", "color": "Reward"},
    text_auto=".0f",
    border=True,
    width=850,
    height=350,
)

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

    value = np.zeros((num_states,))
    transition_matrix = env.T[range(num_states), pi, :]     # shape [s, s_next]
    reward_matrix = env.R[range(num_states), pi, :]         # shape [s, s_next]

    for i in range(max_iterations):
        new_value = einops.reduce(transition_matrix * (reward_matrix + gamma * value), "s s_next -> s", "sum")
        delta = np.abs(new_value - value).max()
        if delta < eps:
            break
        value = np.copy(new_value)
    else:
        print(f"Failed to converge after {max_iterations} steps.")

    return value
    

    
tests.test_policy_eval(policy_eval_numerical, exact=False)
# %%

def policy_eval_exact(env: Environment, pi: Arr, gamma=0.99) -> Arr:
    """
    Finds the exact solution to the Bellman equation.
    """
    num_states = env.num_states

    transition_matrix = env.T[range(num_states), pi, :]
    reward_matrix = env.R[range(num_states), pi, :]

    identity_matrix = np.eye(len(transition_matrix.shape[0]))

    np.linalg.solve()


tests.test_policy_eval(policy_eval_exact, exact=True)