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
section = "part21_dqn"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from gpu_env import CartPole, MountainCar
from gpu_probe import Probe1, Probe2, Probe3, Probe4, Probe5
from rl_utils import ENVS, AtariEnvs, LiveVideo, log_greedy_rollout_video, log_grid_video, make_envs
import part21_dqn.tests as tests
from part1_intro_to_rl.utils import set_global_seeds
from plotly_utils import line, plot_cartpole_obs_and_dones

# The networks in this section are tiny, and if you train on the CPU with
# PyTorch's default of one thread per core, threads spend longer
# coordinating than computing. Adjust if needed.
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


MAIN = __name__ == "__main__"

# %%
envs = CartPole(num_envs=4, device=device)

print(envs.single_action_space)  # 2 actions: left and right
print(envs.single_observation_space)  # Box(4): each observation can take a continuous range of values

obs, infos = envs.reset()
print(obs.shape, obs.device)  # (num_envs, 4): one observation per environment, as a tensor on our device

# %%
class QNetwork(nn.Module):
    """
    For consistency with your tests, please wrap your modules in a `nn.Sequential` called `layers`.
    """

    layers: nn.Sequential

    def __init__(self, obs_shape: tuple[int], num_actions: int, hidden_sizes: list[int] = [120, 84]):
        super().__init__()
        assert len(obs_shape) == 1, "Expecting a single vector of observations"

        layer_list = []
        layer_list.append(nn.Linear(obs_shape[0], hidden_sizes[0]))
        layer_list.append(nn.ReLU())

        for ind in range(len(hidden_sizes) - 1):
            layer_list.append(nn.Linear(hidden_sizes[ind], hidden_sizes[ind + 1]))
            layer_list.append(nn.ReLU())

        layer_list.append(nn.Linear(hidden_sizes[-1], num_actions))

        self.layers = nn.Sequential(*layer_list)


    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


net = QNetwork(obs_shape=(4,), num_actions=2)
n_params = sum((p.nelement() for p in net.parameters()))
assert isinstance(getattr(net, "layers", None), nn.Sequential)
print(net)
print(f"Total number of parameters: {n_params}")
print("You should manually verify network is Linear-ReLU-Linear-ReLU-Linear")
assert not isinstance(net.layers[-1], nn.ReLU)
assert n_params == 10934

# %%
@dataclass
class ReplayBufferSamples:
    """
    Samples from the replay buffer, as PyTorch tensors ready to feed into the Q-network.

    Data is equivalent to (s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1}). Note - here, d_{t+1} is actually **terminated** rather
    than **done** (i.e. it records the times when we went out of bounds, not when the environment timed out).
    """

    obs: Float[Tensor, " sample_size *obs_shape"]
    actions: Int[Tensor, " sample_size"]
    rewards: Float[Tensor, " sample_size"]
    terminated: Bool[Tensor, " sample_size"]
    next_obs: Float[Tensor, " sample_size *obs_shape"]


class ReplayBuffer:
    """
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.
    """

    rng: t.Generator
    obs: Float[Tensor, " buffer_size *obs_shape"]
    actions: Int[Tensor, " buffer_size"]
    rewards: Float[Tensor, " buffer_size"]
    terminated: Bool[Tensor, " buffer_size"]
    next_obs: Float[Tensor, " buffer_size *obs_shape"]

    def __init__(
        self,
        num_envs: int,
        obs_shape: tuple[int],
        action_shape: tuple[int],
        buffer_size: int,
        seed: int,
        device: t.device = device,
    ):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.device = t.device(device)
        self.rng = t.Generator(device=self.device).manual_seed(seed)

        self.obs = t.empty((0, *self.obs_shape), dtype=t.float32, device=self.device)
        self.actions = t.empty((0, *self.action_shape), dtype=t.int64, device=self.device)
        self.rewards = t.empty(0, dtype=t.float32, device=self.device)
        self.terminated = t.empty(0, dtype=t.bool, device=self.device)
        self.next_obs = t.empty((0, *self.obs_shape), dtype=t.float32, device=self.device)

    def add(
        self,
        obs: Float[Tensor, " num_envs *obs_shape"],
        actions: Int[Tensor, " num_envs"],
        rewards: Float[Tensor, " num_envs"],
        terminated: Bool[Tensor, " num_envs"],
        next_obs: Float[Tensor, " num_envs *obs_shape"],
    ) -> None:
        """
        Add a batch of transitions to the replay buffer.
        """
        # Check shapes & datatypes as `t.cat` silently *promotes* dtypes
        for name, data, expected_shape, expected_dtype in zip(
            ["obs", "actions", "rewards", "terminated", "next_obs"],
            [obs, actions, rewards, terminated, next_obs],
            [self.obs_shape, self.action_shape, (), (), self.obs_shape],
            [t.float32, t.int64, t.float32, t.bool, t.float32],
        ):
            assert isinstance(data, Tensor), f"`{name}` should be a tensor, got {type(data)}"
            assert data.shape == (self.num_envs, *expected_shape), f"`{name}` has shape {tuple(data.shape)}, expected {(self.num_envs, *expected_shape)}"
            assert data.dtype == expected_dtype, f"`{name}` has dtype {data.dtype}, expected {expected_dtype}"

        # Add data to buffer, slicing off the old elements
        self.obs = t.cat((self.obs, obs))[-self.buffer_size :]
        self.actions = t.cat((self.actions, actions))[-self.buffer_size :]
        self.rewards = t.cat((self.rewards, rewards))[-self.buffer_size :]
        self.terminated = t.cat((self.terminated, terminated))[-self.buffer_size :]
        self.next_obs = t.cat((self.next_obs, next_obs))[-self.buffer_size :]

    def sample(self, sample_size: int) -> ReplayBufferSamples:
        """
        Sample a batch of transitions from the buffer, with replacement.
        """
        # Independent uniform draws, so duplicates are possible
        indices = t.randint(0, self.obs.shape[0], (sample_size,), generator=self.rng, device=self.device)

        return ReplayBufferSamples(
            obs=self.obs[indices],                  # s_t
            actions=self.actions[indices],          # a_t
            rewards=self.rewards[indices],          # r_{t+1}
            terminated=self.terminated[indices],    # d_{t+1}: terminated (not truncated!)
            next_obs=self.next_obs[indices],        # s_{t+1}
        )



# %%
buffer = ReplayBuffer(num_envs=1, obs_shape=(4,), action_shape=(), buffer_size=256, seed=0, device=device)
envs = CartPole(num_envs=1, seed=0, device=device)
obs, infos = envs.reset()

for i in range(256):
    # Choose random actions (one per environment), and take a step in the environment
    actions = t.randint(0, envs.single_action_space.n, (envs.num_envs,), device=device)
    next_obs, rewards, terminated, truncated, infos = envs.step(actions)

    # Get `true_next_obs` by finding all environments where we terminated & replacing `next_obs`
    # with the actual terminal states (the env has already auto-reset those environments, and stored
    # the observation they ended on in `infos["final_observation"]`)
    done = terminated | truncated
    true_next_obs = next_obs.clone()
    true_next_obs[done] = infos["final_observation"][done]

    # Add experience to buffer
    buffer.add(obs, actions, rewards, terminated, true_next_obs)
    obs = next_obs

sample = buffer.sample(256)

plot_cartpole_obs_and_dones(
    buffer.obs.cpu(),
    buffer.terminated.cpu(),
    title="Current obs s<sub>t</sub><br>so when d<sub>t+1</sub> = 1, these are the states just before termination",
)

plot_cartpole_obs_and_dones(
    buffer.next_obs.cpu(),
    buffer.terminated.cpu(),
    title="Next obs s<sub>t+1</sub><br>so when d<sub>t+1</sub> = 1, these are the terminated states",
)

plot_cartpole_obs_and_dones(
    sample.obs.cpu(),
    sample.terminated.cpu(),
    title="Current obs s<sub>t</sub> (sampled)<br>this is what gets fed into our model for training",
)

# %%
def linear_schedule(
    current_step: int,
    start_e: float,
    end_e: float,
    exploration_fraction: float,
    total_timesteps: int,
) -> float:
    """
    Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction
    * total_timesteps). In other words, we are in "explore mode" with start_e >= epsilon >= end_e
    for the first `exploration_fraction` fraction of total timesteps, and then stay at end_e for the
    rest of training.
    """
    if current_step <= exploration_fraction * total_timesteps:
        b = start_e
        k = (end_e - start_e) / (exploration_fraction * total_timesteps)
        return k * current_step + b
    else:
        return end_e



epsilons = [
    linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
    for step in range(500)
]
line(
    epsilons,
    labels={"x": "steps", "y": "epsilon"},
    title="Probability of random action",
    height=400,
    width=600,
)

tests.test_linear_schedule(linear_schedule)

# %%
def epsilon_greedy_policy(
    envs: CartPole,
    q_network: QNetwork,
    rng: t.Generator,
    obs: Float[Tensor, " num_envs *obs_shape"],
    epsilon: float,
) -> Int[Tensor, " num_envs"]:
    """
    For each environment independently: with probability epsilon take a random action, otherwise take the greedy
    action according to the q_network.

    Inputs:
        envs:       The batched environment we're acting in
        q_network:  The QNetwork used to approximate the Q-value function
        rng:        A torch random number generator
        obs:        The current observation for each environment, shape (num_envs, *obs_shape)
        epsilon:    The probability of taking a random action

    Returns:
        actions:    The sampled action for each environment, shape (num_envs,).
    """
    random_value = t.rand(obs.shape[0], generator=rng, device=obs.device) # (num_envs)

    explore = random_value < epsilon

    greedy_action = q_network(obs).argmax(dim=-1) # (num_envs)

    num_actions = envs.single_action_space.n

    random_action = t.randint(low=0, high=num_actions, size=(obs.shape[0],), generator=rng, device=device) # (num_envs)

    return t.where(explore, random_action, greedy_action)


tests.test_epsilon_greedy_policy(epsilon_greedy_policy)


# %%
env = Probe1(num_envs=4)
assert env.single_observation_space.shape == (1,)
assert env.single_action_space.shape == ()
obs, _ = env.reset()
assert obs.shape == (4, 1)


# %%
@dataclass
class DQNArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "CartPole-gpu"
    num_envs: int = 16 # number of parallel environments to run
    device: str = device

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "DQNCartPole"
    wandb_entity: str | None = None
    video_log_freq: int | None = None # number of optimization steps between wandb video upload
    steps_per_live_video: int | None = 5_000 # number of optimization steps between live video display
    overwrite_video: bool = True # older live videos are overwritten by newer ones if True. Otherwise, they are plotted one after the other.

    # Duration of different phases / buffer memory settings
    total_timesteps: int = 300_000 # total environment steps (summed over all envs), including buffer filling
    steps_per_update: int = 1 # calls to `agent.play_step` (each = num_envs environment steps) per optimization step
    trains_per_target_update: int = 50 # optimization steps between target network updates
    buffer_size: int = 10_000 # replay buffer capacity (environment steps used to fill it before training)

    # Optimization hparams
    batch_size: int = 512
    learning_rate: float = 5e-3

    # RL-specific
    gamma: float = 0.99
    exploration_fraction: float = 0.1
    start_e: float = 1.0
    end_e: float = 0.02

    def __post_init__(self):
        assert self.total_timesteps - self.buffer_size >= self.steps_per_update * self.num_envs
        # Every call to `agent.play_step` advances all `num_envs` environments by one step; we take one optimization
        # step per `steps_per_update` such calls, so this is the total number of optimization steps
        self.total_training_steps = (self.total_timesteps - self.buffer_size) // (self.steps_per_update * self.num_envs)
        self.device = t.device(self.device)


# %%
class DQNAgent:
    """Base Agent class handling the interaction with the environment."""

    def __init__(
        self,
        envs: CartPole,
        buffer: ReplayBuffer,
        q_network: QNetwork,
        start_e: float,
        end_e: float,
        exploration_fraction: float,
        total_timesteps: int,
        rng: t.Generator,
    ):
        self.envs = envs
        self.buffer = buffer
        self.q_network = q_network
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.rng = rng

        self.step = 0  # Tracking number of environment steps taken (summed across all environments)
        self.obs, _ = self.envs.reset()  # Need a starting observation
        self.epsilon = start_e  # Starting value (will be updated in `get_actions`)

    def play_step(self) -> dict:
        """
        Carries out a single interaction step between agent & environment, and adds results to the
        replay buffer.

        Returns `infos` (a dict containing info we will log).
        """
        # Get actions (using self.get_actions rather than randomly sampling like we did in the demo code before)
        actions = self.get_actions(self.obs)
        # Step our environment with these actions
        next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)
        # Add the new experiences to the buffer: remembering that for environments which just finished an episode, next_obs is already the reset observation, 
        # and the true final observation is in infos["final_observation"].
        done = terminated | truncated
        true_next_obs = next_obs.clone()
        true_next_obs[done] = infos["final_observation"][done]
        self.buffer.add(self.obs, actions, rewards, terminated, true_next_obs)
        # Set your new observation as self.obs, ready for the next step
        self.obs = next_obs

        self.step += self.envs.num_envs
        return infos

    def get_actions(self, obs: Float[Tensor, " num_envs *obs_shape"]) -> Int[Tensor, " num_envs"]:
        """
        Samples actions according to the epsilon-greedy policy using the linear schedule for epsilon.
        """
        # Set self.epsilon according to the linear schedule function & the current global step counter
        self.epsilon = linear_schedule(
            current_step=self.step, start_e=self.start_e, end_e=self.end_e, exploration_fraction=self.exploration_fraction, total_timesteps=self.total_timesteps
        )
        # Sample actions according to the epsilon-greedy policy (i.e. using your epsilon_greedy_policy function), and return them
        actions = epsilon_greedy_policy(envs=self.envs, q_network=self.q_network, rng=self.rng, obs=obs, epsilon=self.epsilon)
        return actions


tests.test_agent(DQNAgent)

# %%
def get_episode_data_from_infos(infos: dict) -> dict[str, int | float] | None:
    """
    Helper function: returns a dict of episode data (length, reward and duration, averaged over the
    environments that finished an episode on this step), if at least one
    terminated.
    """
    for final_info in infos.get("final_info", []):
        if final_info is not None and "episode" in final_info:
            return {
                "episode_length": final_info["episode"]["l"].item(),
                "episode_reward": final_info["episode"]["r"].item(),
                "episode_duration": final_info["episode"]["t"].item(),
            }


class DQNTrainer:
    def __init__(self, args: DQNArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.rng = t.Generator(device=args.device).manual_seed(args.seed)
        self.run_name = f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"

        self.envs = make_envs(args.env_id, args.num_envs, args.seed, args.device)

        # Define some basic variables from our environment (note, we assume a single discrete action space)
        num_envs = self.envs.num_envs
        action_shape = self.envs.single_action_space.shape
        num_actions = self.envs.single_action_space.n
        obs_shape = self.envs.single_observation_space.shape
        assert action_shape == ()

        # Create our replay buffer (on the same device as the environment)
        self.buffer = ReplayBuffer(num_envs, obs_shape, action_shape, args.buffer_size, args.seed, args.device)

        # Create our networks & optimizer (target network should be initialized with a copy of the Q-network's weights)
        self.q_network = QNetwork(obs_shape, num_actions).to(args.device)
        self.target_network = QNetwork(obs_shape, num_actions).to(args.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = t.optim.AdamW(self.q_network.parameters(), lr=args.learning_rate)

        # Create our agent
        self.agent = DQNAgent(
            self.envs,
            self.buffer,
            self.q_network,
            args.start_e,
            args.end_e,
            args.exploration_fraction,
            args.total_timesteps,
            self.rng,
        )

    def add_to_replay_buffer(self, n: int, verbose: bool = False):
        """
        Calls `agent.play_step` n times (n * num_envs environment steps), adding to the replay buffer.
        Should return a dict of data from the last terminated episode, if any (the training loop takes
        care of logging it).

        Optional argument `verbose`: if True, we can use a progress bar (useful to check how long
        the initial buffer filling is taking).
        """
        # This calls self.agent.play_step() n times (i.e. n * num_envs environment steps), which adds the results to the replay buffer
        data = None
        for i in range(n):
            infos = self.agent.play_step()
            new_data = get_episode_data_from_infos(infos)
            if new_data is not None:
                data = new_data
        return data
        # It's used to fill the buffer before training starts, and before each optimization step to add new experiences to the buffer

    def prepopulate_replay_buffer(self):
        """
        Called to fill the replay buffer before training starts.
        """
        n_steps_to_fill_buffer = self.args.buffer_size // self.args.num_envs
        self.add_to_replay_buffer(n_steps_to_fill_buffer, verbose=True)

    def training_step(self, step: int) -> None:
        """
        Samples once from the replay buffer, and takes a single optimization step.

        Args:
            step (int): The number of optimization steps taken so far (used for logging, and for deciding
            when to update the target network)
        """
        

        batch = self.buffer.sample(sample_size=self.args.batch_size)
        #  return ReplayBufferSamples(
        #     obs=self.obs[indices],                  # s_t
        #     actions=self.actions[indices],          # a_t
        #     rewards=self.rewards[indices],          # r_{t+1}
        #     terminated=self.terminated[indices],    # d_{t+1}: terminated (not truncated!)
        #     next_obs=self.next_obs[indices],        # s_{t+1}
        # )

        predicted_q_values = self.q_network(batch.obs) # (batch_size, actions)

        predicted_value = predicted_q_values[range(len(batch.actions)), batch.actions] # (batch_size, )
        with t.inference_mode():
            target_q_values = self.target_network(batch.next_obs)
            target_value = target_q_values.max(dim=-1).values # (batch_size)


        mask =  1 - batch.terminated.float() 
        masked =  mask * self.args.gamma * target_value 
        y_true = batch.rewards + masked
        loss_vector = (y_true - predicted_value) ** 2
        loss = loss_vector.mean(dim=0)

        
        loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()

        if step % self.args.trains_per_target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.args.use_wandb and step % 10 == 0:
            wandb.log(
                {
                    "td_loss": loss,
                    "q_values": predicted_value.mean().item(),
                    "epsilon": self.agent.epsilon,
                },
                step=self.agent.step,
            )


    def train(self) -> None:
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
            )

        self.prepopulate_replay_buffer()

        pbar = tqdm(range(self.args.total_training_steps))
        last_logged_time, last_logged_step = time.time(), self.agent.step  # so we don't log too often

        greedy = lambda obs: self.q_network(obs).argmax(-1)
        fresh_envs = lambda seed: CartPole(num_envs=16, seed=seed, device=self.args.device)
        can_render = self.args.env_id == "CartPole-gpu"
        live_videos = can_render and self.args.steps_per_live_video is not None
        wandb_videos = can_render and self.args.use_wandb and self.args.video_log_freq is not None
        video_slot = LiveVideo(overwrite=self.args.overwrite_video)  # one display slot per run (see rl_utils.LiveVideo)

        for step in pbar:
            data = self.add_to_replay_buffer(self.args.steps_per_update)
            if data is not None:
                if time.time() - last_logged_time > 0.5:
                    now = time.time()
                    if self.args.use_wandb:
                        sps = (self.agent.step - last_logged_step) / (now - last_logged_time)
                        wandb.log({**data, "SPS": sps}, step=self.agent.step)
                    last_logged_time, last_logged_step = now, self.agent.step
                    pbar.set_postfix(**data)

            self.training_step(step)

            live = live_videos and step % self.args.steps_per_live_video == 0
            to_wandb = wandb_videos and step % self.args.video_log_freq == 0
            if live or to_wandb:
                log_greedy_rollout_video(fresh_envs(step), greedy, self.envs.draw, step=self.agent.step, live=live, slot=video_slot)


        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()


# %%
def test_probe(probe_idx: int):
    """
    Tests a probe environment by training a network on it & verifying that the value functions are
    in the expected range.
    """
    # Train our network on this probe env.
    args = DQNArgs(
        env_id=f"Probe{probe_idx}-v0",
        wandb_project_name=f"test-probe-{probe_idx}",
        total_timesteps=3000 if probe_idx <= 2 else 5000,
        learning_rate=0.001,
        buffer_size=500,
        use_wandb=False,
        trains_per_target_update=20,
        video_log_freq=None,
        num_envs=1,
        steps_per_update=10,
        exploration_fraction=0.2,
        end_e=0.1,
    )
    trainer = DQNTrainer(args)
    trainer.train()

    # Get the correct set of observations, and corresponding values we expect
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [
        [[1.0]],
        [[-1.0], [+1.0]],
        [[args.gamma], [1.0]],
        [[-1.0, 1.0]],
        [[1.0, -1.0], [-1.0, 1.0]],
    ]
    tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
    obs = t.tensor(obs_for_probes[probe_idx - 1]).to(device)

    # Calculate the actual value, and verify it
    value = trainer.q_network(obs)
    # A bf16 network (e.g. under autocast) can't represent these targets to better than its eps
    atol = max(tolerances[probe_idx - 1], t.finfo(value.dtype).eps)
    expected_value = t.tensor(expected_value_for_probes[probe_idx - 1]).to(device)
    t.testing.assert_close(value.float(), expected_value, atol=atol, rtol=0)
    print("Probe tests passed!\n")


for probe_idx in range(1, 6):
    test_probe(probe_idx)

# %%
args = DQNArgs(use_wandb=True)
trainer = DQNTrainer(args)
trainer.train()


# %%
# Re-run training with the default settings, but every 500 gradient steps record four things:
#   1. how long the greedy policy survives on 16 fresh environments (no exploration)
#   2. how many of the transitions in the replay buffer are terminal, i.e. does the data still contain failures
#   3. the action gap  max_a Q(s,a) - min_a Q(s,a)  averaged over a fixed batch of states from the random-policy phase
#   4. max_a Q(s, a) on a fixed set of doomed states, from which every action terminates next step (correct value 0)
#   5. the TD loss, recomputed on a large sample from the buffer (the same quantity training_step minimises)
from plotly.subplots import make_subplots
from rl_utils import greedy_rollout

# toggle this back to true to run this
RUN_ACTION_GAP_DEMO = False

if RUN_ACTION_GAP_DEMO:
    args = DQNArgs(use_wandb=False, steps_per_live_video=None)
    trainer = DQNTrainer(args)
    trainer.prepopulate_replay_buffer()
    probe_obs = trainer.buffer.sample(2048).obs  # sampled now, while the buffer is still full of random-policy failures
    pre_terminal_obs = trainer.buffer.obs[trainer.buffer.terminated]  # states whose taken action ended the episode
    # Check these are doomed under BOTH actions, by writing each state into a batched env and stepping it with each action.
    check_env = CartPole(num_envs=len(pre_terminal_obs), device=args.device)
    doomed = t.ones(len(pre_terminal_obs), dtype=t.bool, device=args.device)
    for a in (0, 1):
        check_env.state = pre_terminal_obs.clone()
        doomed &= check_env.step(t.full((len(pre_terminal_obs),), a, device=args.device))[2]  # terminated flag
    print(f"{doomed.sum().item()} of {len(doomed)} pre-termination states are doomed under both actions")
    doomed_obs = pre_terminal_obs[doomed]
    greedy = lambda obs: trainer.q_network(obs).argmax(-1)

    def td_loss(n_samples: int = 2048) -> float:
        d = trainer.buffer.sample(n_samples)
        with t.inference_mode():
            target = d.rewards + args.gamma * trainer.target_network(d.next_obs).max(-1).values * (1 - d.terminated.float())
            predicted = trainer.q_network(d.obs).gather(-1, d.actions.unsqueeze(-1)).squeeze(-1)
        return (target - predicted).pow(2).mean().item()

    log = []
    pbar = tqdm(range(args.total_training_steps))
    for step in pbar:
        trainer.add_to_replay_buffer(args.steps_per_update)
        trainer.training_step(step)
        if (step + 1) % 500 == 0:
            _, _, lifespan = greedy_rollout(CartPole(num_envs=16, seed=step, device=args.device), greedy)
            with t.inference_mode():
                q = trainer.q_network(probe_obs)
                q_doomed = trainer.q_network(doomed_obs).max(-1).values
            record = {
                "G" : lifespan.mean().item(),
                "n_term" : trainer.buffer.terminated.sum().item(),
                "act_gap" : (q.max(-1).values - q.min(-1).values).mean().item(),
                "q_doom" : q_doomed.mean().item(),
                "loss" : td_loss(),
            }
            log.append([step + 1, *record.values()])
            pbar.set_postfix({k: f"{v:.3g}" for k, v in record.items()})
else:
    print(f"{RUN_ACTION_GAP_DEMO=} Skipping action gap demo.")


# %%
