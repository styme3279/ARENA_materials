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

    value = np.zeros(num_states)
    transition_matrix = env.T[range(num_states), pi, :]  # shape [s, s_next]
    reward_matrix = env.R[range(num_states), pi, :]  # shape [s, s_next]

    num_iterations = 0
    delta = 2*eps
    while num_iterations <= max_iterations and delta >= eps:
        new_value = einops.reduce(transition_matrix * (reward_matrix + gamma * value[None, :]), "s s_next -> s", "sum")
        delta = np.abs(new_value - value).max()
        value = np.copy(new_value)
        num_iterations += 1
    return value

tests.test_policy_eval(policy_eval_numerical, exact=False)

# %%
def policy_eval_exact(env: Environment, pi: Arr, gamma=0.99) -> Arr:
    """
    Finds the exact solution to the Bellman equation.
    """
    num_states = env.num_states
    transition_matrix = env.T[range(num_states), pi, :]  # shape [s, s_next]
    reward_matrix = env.R[range(num_states), pi, :]  # shape [s, s_next]
    rewards = einops.einsum(transition_matrix, reward_matrix, "s s_next, s s_next -> s")
    mat = (np.identity(num_states) - gamma * transition_matrix)
    return np.linalg.solve(mat, rewards)

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
    # Q: [num_states, action]
    T = env.T # [num_states, action, num_states]
    R = env.R # [num_states, action, num_states]
    Q = einops.reduce(T * (R + gamma * V), "s action s_new -> s action", "sum")
    return np.argmax(Q, axis=-1)


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
        values = policy_eval_exact(env=env, gamma=gamma, pi=pi)
        pi = policy_improvement(env=env, V=values, gamma=gamma)
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
        self.policy = find_optimal_policy(env.unwrapped.env, gamma)

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
        # with prob 1-eps: self.Q[obs].argmax()
        # with prob eps: random
        if self.rng.random() < self.config.epsilon:
            return self.rng.integers(0, self.num_actions)
        else:
            return self.Q[obs].argmax()

class QLearning(EpsilonGreedy):
    def observe(self, exp: Experience) -> None:
        state, action, reward, next_state = exp.obs, exp.act, exp.reward, exp.new_obs
        self.Q[state][action] += self.config.lr * (reward + (self.gamma * self.Q[next_state].max()) - self.Q[state][action])

class SARSA(EpsilonGreedy):
    def observe(self, exp: Experience):
        state, action, reward, next_state, next_action = exp.obs, exp.act, exp.reward, exp.new_obs, exp.new_act
        self.Q[state][action] += self.config.lr * (reward + (self.gamma * self.Q[next_state, next_action]) - self.Q[state][action])

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
n_seeds = 5
norvig_specs = [
    (Cheater, AgentConfig(), "Cheater"),
    (QLearning, AgentConfig(), "Q-Learning"),
    (SARSA, AgentConfig(), "SARSA"),
    (Random, AgentConfig(), "Random"),
]
# average over a few seeds (tabular RL is noisy) and plot mean ± standard error
curves = utils.seeded_curves(norvig_specs, "NorvigGrid-v0", gamma=gamma, n_runs=n_runs, n_seeds=n_seeds)
title = f"Avg. reward on NorvigGrid-v0 (mean ± se, {n_seeds} seeds)"
utils.plot_learning_curves(curves, title=title, yaxis_title="Avg. reward", band="se")

# %%
explore_grid = GridWorld("....T\n"
                        "...G.\n"
                        ".....\n"
                        ".....\n"
                        "S....")
explore_grid.show_map(show=True, title="Deterministic Exploration World")

gym.envs.registration.register(
    id="ExplorationGrid-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=100,
    nondeterministic=False,
    kwargs={"env": explore_grid},
)

# %%
gamma = 0.99
n_runs = 100
n_seeds = 5
epsilons = [0.0, 0.1, 0.2, 0.5]

# one Q-learning agent per epsilon, averaged over several seeds (tabular RL is noisy), plotted mean ± se
explore_specs = [(QLearning, AgentConfig(epsilon=eps, lr=0.1, optimism=0.0), f"epsilon={eps}") for eps in epsilons]
curves = utils.seeded_curves(explore_specs, "ExplorationGrid-v0", gamma=gamma, n_runs=n_runs, n_seeds=n_seeds)
title = f"Q-learning on ExplorationGrid-v0 (mean ± se, {n_seeds} seeds)"
utils.plot_learning_curves(curves, title=title, band="se")

# %%
@dataclass
class TD_LambdaConfig(AgentConfig):
    lambda_: float = 0.95

class SARSA_lambda(SARSA):
    def __init__(
        self,
        env: DiscreteEnviroGym,
        config: TD_LambdaConfig = defaultConfig,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        super().__init__(env, config, gamma, seed)
        self.elegibility_trace = np.zeros((self.num_states, self.num_actions))

    def run_episode(self, seed) -> list[float]:
        self.elegibility_trace = np.zeros((self.num_states, self.num_actions))
        return super().run_episode(seed)

    def observe(self, exp: Experience):
        state, action, reward, next_state, next_action = exp.obs, exp.act, exp.reward, exp.new_obs, exp.new_act
        delta = reward + (self.gamma * self.Q[next_state, next_action]) - self.Q[state][action]
        self.elegibility_trace[state][action] += 1
        self.Q += self.config.lr * delta * self.elegibility_trace
        self.elegibility_trace *= self.gamma * self.config.lambda_

# %%
gamma = 0.99
n_runs = 400
n_seeds = 5
norvig_lambda_specs = [
    (Cheater, AgentConfig(), "Cheater"),
    (QLearning, AgentConfig(), "Q-Learning"),
    (SARSA, AgentConfig(), "SARSA"),
    (SARSA_lambda, TD_LambdaConfig(), "SARSA(λ)"),
    (Random, AgentConfig(), "Random"),
]
# average over a few seeds (tabular RL is noisy) and plot mean ± standard error
curves = utils.seeded_curves(norvig_lambda_specs, "NorvigGrid-v0", gamma=gamma, n_runs=n_runs, n_seeds=n_seeds)
title = f"Avg. reward on NorvigGrid-v0 with SARSA(λ) (mean ± se, {n_seeds} seeds)"
utils.plot_learning_curves(curves, title=title, yaxis_title="Avg. reward", band="se")

# %%
gym.envs.registration.register(
    id="LargeGrid-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=200,
    nondeterministic=False,
    kwargs={"env": GridWorld(".......G\n........\n........\n........\n........\n........\n........\nS.......")},
)

gamma = 0.99
n_runs = 150
n_seeds = 5
agent_specs = [
    (SARSA, AgentConfig(epsilon=0.2, lr=0.2), "SARSA (1-step)", (99, 110, 250)),
    (SARSA_lambda, TD_LambdaConfig(epsilon=0.2, lr=0.2, lambda_=0.8), r"SARSA($\lambda$=0.8)", (239, 85, 59)),
]


# average over several seeds (tabular RL is noisy) and plot mean ± se, reusing each spec's colour
curves = utils.seeded_curves([(Cls, cfg, name) for Cls, cfg, name, _ in agent_specs],
                             "LargeGrid-v0", gamma=gamma, n_runs=n_runs, n_seeds=n_seeds)
title = f"Eligibility traces learn faster on a larger gridworld (8×8, sparse reward; mean ± se, {n_seeds} seeds)"
utils.plot_learning_curves(curves, title=title, band="se", colors=[color for *_, color in agent_specs])

# %%
gamma = 1
n_runs = 500
config_cliff = AgentConfig(epsilon=0.1, lr=0.1, optimism=0)
cliff_specs = [(QLearning, config_cliff, "Q-Learning"), (SARSA, config_cliff, "SARSA")]

# train one agent of each type (kept in `agents` for the greedy-rollout section below), show its
# learned policy, and collect its cumulative-mean return. The curves are near-identical across seeds
# here, so a single run is plenty - no need to average.
agents = []
returns_list = []
for AgentCls, config, name in cliff_specs:
    agent = AgentCls(gym.make("CliffWalking-v1", is_slippery=False), config, gamma, seed=0)
    returns = agent.train(n_runs)
    agents.append(agent)
    returns_list.append(utils.cummean(returns))
    V = agent.Q.max(axis=-1).reshape(4, 12)
    pi = agent.Q.argmax(axis=-1).reshape(4, 12)
    cliffwalk_imshow(V, pi, title=f"CliffWalking: {name} Agent", width=800, height=400)

# rewards are always negative; plot the magnitude on a reversed log y-axis with a "-" tick prefix,
# so the axis reads as negative reward (closer to 0 = better = higher)
fig = line(
    [-r for r in returns_list],
    names=[name for _, _, name in cliff_specs],
    template="simple_white",
    title="Q-Learning vs SARSA on CliffWalking-v1",
    labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
    log_y=True,
    width=700,
    height=400,
    return_fig=True,
)
fig.update_layout(yaxis=dict(autorange="reversed", tickprefix="-"))
fig.show()

# %%
def greedy_return(env: gym.Env, Q: Arr, max_steps: int = 200) -> float:
    """Roll out the greedy (epsilon=0) policy implied by Q, and return its total reward."""
    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(int(Q[obs].argmax()))
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


for agent in agents:
    ret = greedy_return(gym.make("CliffWalking-v1", is_slippery=False), agent.Q)
    print(f"{agent.name:10s} greedy-policy return = {ret:.0f}")

# %%
class CliffWalking(Environment):
    def __init__(self, penalty=-1):
        self.height = 4
        self.width = 12
        self.penalty = penalty
        num_states = self.height * self.width
        num_actions = 4
        self.states = np.array([[x, y] for y in range(self.height) for x in range(self.width)])
        self.actions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])  # up, right, down, left
        self.dim = (self.height, self.width)

        # special states: tuples of state and reward
        # all other states get penalty
        start = 36
        terminal = np.array([47], dtype=int)
        self.cliff = np.arange(37, 47, dtype=int)
        self.goal_rewards = np.array([1.0, -1.0])

        super().__init__(num_states, num_actions, start=start, terminal=terminal)

    def dynamics(self, state: int, action: int) -> tuple[Arr, Arr, Arr]:
        """
        Returns tuple of (out_states, out_rewards, out_probs) for this given (state, action) pair.
        """

        def state_index(state):
            assert 0 <= state[0] < self.width and 0 <= state[1] < self.height, print(state)
            pos = state[0] + state[1] * self.width
            assert 0 <= pos < self.num_states, print(state, pos)
            return pos

        pos = self.states[state]

        if state in self.terminal:
            return (np.array([state]), np.array([0]), np.array([1]))

        # No slipping; each action is deterministic
        out_probs = np.zeros(self.num_actions)
        out_probs[action] = 1

        out_states = np.zeros(self.num_actions, dtype=int) + self.num_actions
        out_rewards = np.zeros(self.num_actions) + self.penalty
        new_states = [pos + x for x in self.actions]

        for i, s_new in enumerate(new_states):
            if not (0 <= s_new[0] < self.width and 0 <= s_new[1] < self.height):
                out_states[i] = state
                continue

            new_state = state_index(s_new)

            # Check if would hit the cliff, if so then get -100 penalty and go back to start
            if new_state in self.cliff:
                out_states[i] = self.start
                out_rewards[i] -= 100

            else:
                out_states[i] = new_state

            for idx in range(len(self.terminal)):
                if new_state == self.terminal[idx]:
                    out_rewards[i] = self.goal_rewards[idx]

        return (out_states, out_rewards, out_probs)

    @staticmethod
    def render(Q: Arr, name: str):
        V = Q.max(axis=-1).reshape(4, 12)
        pi = Q.argmax(axis=-1).reshape(4, 12)
        cliffwalk_imshow(V, pi, title=f"CliffWalking: {name} Agent")

gym.envs.registration.register(
    id="CliffWalking-myversion",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=200,
    nondeterministic=True,
    kwargs={"env": CliffWalking(penalty=-1)},
)
gamma = 0.99
seed = 0
config_cliff = AgentConfig(epsilon=0.1, lr=0.1, optimism=0)
env = gym.make("CliffWalking-myversion")
n_runs = 500
args_cliff = (env, config_cliff, gamma, seed)

agents = [Cheater(*args_cliff), QLearning(*args_cliff), SARSA(*args_cliff), Random(*args_cliff)]
returns_list = []
name_list = []

for agent in agents:
    returns = agent.train(n_runs)[1:]
    returns_list.append(utils.cummean(returns))
    name_list.append(agent.name)

line(
    returns_list,
    names=name_list,
    template="simple_white",
    title="Q-Learning vs SARSA on CliffWalking-v1",
    labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
    width=700,
    height=400,
)

# %%
