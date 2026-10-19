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
ObsType: TypeAlias = int | np.ndarray
ActType: TypeAlias = int


class DiscreteEnviroGym(gym.Env):
    """
    A discrete environment class for reinforcement learning, compatible with OpenAI Gym.

    This class represents a discrete environment where actions and observations are discrete.
    It is designed to interface with a provided `Environment` object which defines the
    underlying dynamics, states, and actions.

    Attributes:
        action_space (gym.spaces.Discrete): The space of possible actions.
        observation_space (gym.spaces.Discrete): The space of possible observations (states).
        env (Environment): The underlying environment with its own dynamics and properties.
    """

    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete

    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.observation_space = gym.spaces.Discrete(env.num_states)
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        """
        Execute an action and return the new state, reward, terminated flag, truncated flag, and additional info.
        The behaviour of this function depends primarily on the dynamics of the underlying
        environment.
        """
        states, rewards, probs = self.env.dynamics(self.pos, action)
        idx = self.np_random.choice(len(states), p=probs)
        new_state, reward = states[idx], rewards[idx]
        self.pos = new_state
        terminated = self.pos in self.env.terminal
        truncated = False
        info = {"env": self.env}
        return new_state, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options=None) -> tuple[ObsType, dict]:
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        self.pos = self.env.start
        return self.pos, {}

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"
# %%
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
gym.envs.registration.register(
    id="NorvigGrid-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=100,
    nondeterministic=True,
    kwargs={"env": Norvig(penalty=-0.04)},
)

gym.envs.registration.register(
    id="ToyGym-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=3,  # use 3 not 2, because of 1-indexing
    nondeterministic=False,
    kwargs={"env": Toy()},
)
# %%
@dataclass
class Experience:
    """
    A class for storing one piece of experience during an episode run.
    """

    obs: ObsType
    act: ActType
    reward: float
    new_obs: ObsType
    new_act: ActType | None = None


@dataclass
class AgentConfig:
    """Hyperparameters for agents"""

    epsilon: float = 0.1
    lr: float = 0.05
    optimism: float = 0


defaultConfig = AgentConfig()


class Agent:
    """
    Base class for agents interacting with an environment.

    You do not need to add any implementation here.
    """

    rng: np.random.Generator

    def __init__(
        self,
        env: DiscreteEnviroGym,
        config: AgentConfig = defaultConfig,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        self.env = env
        self.reset(seed)
        self.config = config
        self.gamma = gamma
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        self.name = type(self).__name__

    def get_action(self, obs: ObsType) -> ActType:
        raise NotImplementedError()

    def observe(self, exp: Experience) -> None:
        """
        Agent observes experience, and updates model as appropriate.
        Implementation depends on type of agent.
        """
        pass

    def reset(self, seed: int) -> tuple[ObsType, dict]:
        self.rng = np.random.default_rng(seed)
        return None, {}

    def run_episode(self, seed) -> list[int]:
        """
        Simulates one episode of interaction, agent learns as appropriate

        Inputs:
            seed : Seed for the random number generator

        Returns:
            The rewards obtained during the episode
        """
        rewards = []
        obs, info = self.env.reset(seed=seed)
        self.reset(seed=seed)
        done = False
        while not done:
            act = self.get_action(obs)
            new_obs, reward, terminated, truncated, info = self.env.step(act)
            done = terminated or truncated
            exp = Experience(obs, act, reward, new_obs)
            self.observe(exp)
            rewards.append(reward)
            obs = new_obs
        return rewards

    def train(self, n_runs=500):
        """
        Run a batch of episodes, and return the total reward obtained per episode

        Inputs:
            n_runs : The number of episodes to simulate

        Returns:
            The discounted sum of rewards obtained for each episode
        """
        all_returns = []
        for seed in range(n_runs):
            rewards = self.run_episode(seed)
            all_returns.append(utils.sum_rewards(rewards, self.gamma))
        return all_returns


class Random(Agent):
    def get_action(self, obs: ObsType) -> ActType:
        return self.rng.integers(0, self.num_actions)
# %%
class Cheater(Agent):
    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma=0.99, seed=0):
        super().__init__(env, config, gamma, seed)
        self.policy = find_optimal_policy(self.env.unwrapped.env, gamma=gamma)

    def get_action(self, obs):
        return self.policy[obs]


env_toy = gym.make("ToyGym-v0")
agents_toy: list[Agent] = [Cheater(env_toy), Random(env_toy)]
returns_dict = {}
for agent in agents_toy:
    returns = agent.train(n_runs=100)
    returns_dict[agent.name] = utils.cummean(returns)

line(
    list(returns_dict.values()),
    names=list(returns_dict.keys()),
    title=f"Avg. reward on {env_toy.spec.name}",
    labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
    template="simple_white",
    width=700,
    height=400,
)
# %%
class EpsilonGreedy(Agent):
    """
    A class for SARSA and Q-Learning to inherit from.
    """

    def __init__(
        self,
        env: DiscreteEnviroGym,
        config: AgentConfig = defaultConfig,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        super().__init__(env, config, gamma, seed)
        self.Q = np.zeros((self.num_states, self.num_actions)) + self.config.optimism
        # Terminal states have value 0. We never act from them so their Q-values are never updated, but
        # Q-learning & SARSA bootstrap from them, so they mustn't keep the optimistic initial value.
        if hasattr(env.unwrapped, "env"):  # our `DiscreteEnviroGym` envs expose their terminal states
            self.Q[env.unwrapped.env.terminal] = 0

    def get_action(self, obs: ObsType) -> ActType:
        """
        Selects an action using epsilon-greedy with respect to Q-value estimates
        """
        if self.rng.random() <= self.config.epsilon:
            return self.rng.integers(0, self.num_actions)
        else:
            return int(np.argmax(self.Q[obs]))


class QLearning(EpsilonGreedy):
    def observe(self, exp: Experience) -> None:
        obs, act, reward, new_obs = exp.obs, exp.act, exp.reward, exp.new_obs # we dont need actual new_act for Q-learning
        action = np.argmax(self.Q[new_obs])
        # print(f"Updating Q[{obs}, {act}] with reward {reward} and next state {new_obs} (best action {action})")
        self.Q[obs, act] += self.config.lr * (reward + self.gamma * self.Q[new_obs, action] - self.Q[obs, act])



class SARSA(EpsilonGreedy):
    def observe(self, exp: Experience):
        obs, act, reward, new_obs, new_act = exp.obs, exp.act, exp.reward, exp.new_obs, exp.new_act
        self.Q[obs, act] += self.config.lr * (reward + self.gamma * self.Q[new_obs, new_act] - self.Q[obs, act])

    def run_episode(self, seed) -> list[float]:
        rewards = []
        obs, info = self.env.reset(seed=seed)
        act = self.get_action(obs)
        self.reset(seed=seed)
        done = False
        while not done:
            new_obs, reward, terminated, truncated, info = self.env.step(act)
            done = terminated or truncated
            new_act = self.get_action(new_obs)
            exp = Experience(obs, act, reward, new_obs, new_act)
            self.observe(exp)
            rewards.append(reward)
            obs = new_obs
            act = new_act
        return rewards


gamma = 0.99
n_runs = 400
n_seeds = 1
norvig_specs = [
    (Cheater, AgentConfig(), "Cheater"),
    (QLearning, AgentConfig(), "Q-Learning"),
    # (SARSA, AgentConfig(), "SARSA"),
    # (Random, AgentConfig(), "Random"),
]
# average over a few seeds (tabular RL is noisy) and plot mean ± standard error
curves = utils.seeded_curves(norvig_specs, "NorvigGrid-v0", gamma=gamma, n_runs=n_runs, n_seeds=n_seeds)
print(f"Curves: {curves.keys()}")
title = f"Avg. reward on NorvigGrid-v0 (mean ± se, {n_seeds} seeds)"
utils.plot_learning_curves(curves, title=title, yaxis_title="Avg. reward", band="se")
# plot the difference between Q-learning and Cheater policy

# %%
for lr in [0.01, 0.05, 0.1, 0.2, 0.5]:
    norvig_specs = [
        (QLearning, AgentConfig(lr=lr), f"Q-Learning (lr={lr})"),
        (SARSA, AgentConfig(lr=lr), f"SARSA (lr={lr})"),
    ]
    curves = utils.seeded_curves(norvig_specs, "NorvigGrid-v0", gamma=gamma, n_runs=n_runs, n_seeds=n_seeds)
    title = f"Avg. reward on NorvigGrid-v0 (mean ± se, {n_seeds} seeds)"
    utils.plot_learning_curves(curves, title=title, yaxis_title="Avg. reward", band="se")