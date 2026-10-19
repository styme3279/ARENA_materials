# %%
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import torch as t
from IPython.display import HTML, display
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part54_toy_models_of_superposition_and_saes"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part54_toy_models_of_superposition_and_saes.tests as tests
import part54_toy_models_of_superposition_and_saes.utils as utils
from plotly_utils import imshow, line

MAIN = __name__ == "__main__"

# %%
t.manual_seed(2)

W = t.randn(2, 5)
W_normed = W / W.norm(dim=0, keepdim=True)

imshow(
    W_normed.T @ W_normed,
    title="Cosine similarities of each pair of 2D feature embeddings",
    width=600,
)

# %%
utils.plot_features_in_2d(
    W_normed.unsqueeze(0),  # shape [instances=1 d_hidden=2 features=5]
)

# %%
def linear_lr(step, steps):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


@dataclass
class ToyModelConfig:
    # We optimize n_inst models in a single training loop to let us sweep over sparsity or importance
    # curves efficiently. You should treat the number of instances `n_inst` like a batch dimension,
    # but one which is built into our training setup. Ignore the latter 3 arguments for now, they'll
    # return in later exercises.
    n_inst: int
    n_features: int = 5
    d_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    feat_mag_distn: Literal["unif", "normal"] = "unif"


class ToyModel(nn.Module):
    W: Float[Tensor, "inst d_hidden feats"]
    b_final: Float[Tensor, "inst feats"]

    # Our linear map (for a single instance) is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: ToyModelConfig,
        feature_probability: float | Tensor = 0.01,
        importance: float | Tensor = 1.0,
        device=device,
    ):
        super(ToyModel, self).__init__()
        self.cfg = cfg

        if isinstance(feature_probability, float):
            feature_probability = t.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to((cfg.n_inst, cfg.n_features))
        if isinstance(importance, float):
            importance = t.tensor(importance)
        self.importance = importance.to(device).broadcast_to((cfg.n_inst, cfg.n_features))

        self.W = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_inst, cfg.d_hidden, cfg.n_features))))
        self.b_final = nn.Parameter(t.zeros((cfg.n_inst, cfg.n_features)))
        self.to(device)

    def forward(
        self,
        features: Float[Tensor, "... inst feats"],
    ) -> Float[Tensor, "... inst feats"]:
        """
        Performs a single forward pass. For a single instance, this is given by:
            x -> ReLU(W.T @ W @ x + b_final)
        """
        h = einops.einsum(
            features, self.W, "... inst feats, inst hidden feats -> ... inst hidden"
        )
        out = einops.einsum(
            h, self.W, "... inst hidden, inst hidden feats -> ... inst feats"
        )
        return F.relu(out + self.b_final)


    def generate_batch(self, batch_size: int) -> Float[Tensor, "batch inst feats"]:
        """
        Generates a batch of data of shape (batch_size, n_inst, n_features).
        """
        # You'll fill this in later
        raise NotImplementedError()

    def calculate_loss(
        self,
        out: Float[Tensor, "batch inst feats"],
        batch: Float[Tensor, "batch inst feats"],
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss for a given batch (as a scalar tensor), using the loss described in the
        Toy Models of Superposition paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Note, `self.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
        """
        # You'll fill this in later
        raise NotImplementedError()

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 5_000,
        log_freq: int = 50,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        """
        Optimizes the model using the given hyperparameters.
        """
        optimizer = t.optim.Adam(self.parameters(), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:
            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(loss=loss.item() / self.cfg.n_inst, lr=step_lr)


tests.test_model(ToyModel)

# %%
def generate_batch(self: ToyModel, batch_size: int) -> Float[Tensor, "batch inst feats"]:
    """
    Generates a batch of data of shape (batch_size, n_inst, n_features).
    """
    batch_shape = (batch_size, self.cfg.n_inst, self.cfg.n_features)
    feat_mag = t.rand(batch_shape, device=self.W.device)
    feat_seeds = t.rand(batch_shape, device=self.W.device)
    return t.where(feat_seeds <= self.feature_probability, feat_mag, 0.0)


ToyModel.generate_batch = generate_batch
tests.test_generate_batch(ToyModel)


# %%
def calculate_loss(
    self: ToyModel,
    out: Float[Tensor, "batch inst feats"],
    batch: Float[Tensor, "batch inst feats"],
) -> Float[Tensor, ""]:
    """
    Calculates the loss for a given batch, using the loss described in the Toy Models paper:

        https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

    Remember, `self.importance` will always have shape (n_inst, n_features).
    """
    error = self.importance * ((batch - out) ** 2)
    loss = einops.reduce(error, "batch inst feats -> inst", "mean").sum()
    return loss


ToyModel.calculate_loss = calculate_loss
tests.test_calculate_loss(ToyModel)


# %%
cfg = ToyModelConfig(n_inst=8, n_features=5, d_hidden=2)

# importance varies within features for each instance
importance = 0.9 ** t.arange(cfg.n_features)

# sparsity is the same for all features in a given instance, but varies over instances
feature_probability = 50 ** -t.linspace(0, 1, cfg.n_inst)

line(
    importance,
    width=600,
    height=400,
    title="Importance of each feature (same over all instances)",
    labels={"y": "Feature importance", "x": "Feature"},
)
line(
    feature_probability,
    width=600,
    height=400,
    title="Feature probability (varied over instances)",
    labels={"y": "Probability", "x": "Instance"},
)

model = ToyModel(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None],
)
model.optimize()


utils.plot_features_in_2d(
    model.W,
    colors=model.importance,
    title=f"Superposition: {cfg.n_features} features represented in 2D space",
    subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
)

# %%
with t.inference_mode():
    batch = model.generate_batch(200)
    hidden = einops.einsum(
        batch,
        model.W,
        "batch instances features, instances hidden features -> instances hidden batch",
    )

utils.plot_features_in_2d(hidden, title="Hidden state representation of a random batch of data")

# %%
cfg = ToyModelConfig(n_inst=10, n_features=100, d_hidden=20)

importance = 100 ** -t.linspace(0, 1, cfg.n_features)
feature_probability = 20 ** -t.linspace(0, 1, cfg.n_inst)

line(
    importance,
    width=600,
    height=400,
    title="Feature importance (same over all instances)",
    labels={"y": "Importance", "x": "Feature"},
)
line(
    feature_probability,
    width=600,
    height=400,
    title="Feature probability (varied over instances)",
    labels={"y": "Probability", "x": "Instance"},
)

model = ToyModel(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None],
)
model.optimize(steps=10_000)

# %%
utils.plot_features_in_Nd(
    model.W,
    height=800,
    width=1600,
    title="ReLU output model: n_features = 100, d_hidden = 20, I<sub>i</sub> = 0.9<sup>i</sup>",
    subplot_titles=[f"Feature prob = {i:.3f}" for i in feature_probability],
)

# %%
def generate_correlated_features(
    self: ToyModel, batch_size: int, n_correlated_pairs: int
) -> Float[Tensor, "batch inst 2*n_correlated_pairs"]:
    """
    Generates a batch of correlated features. For each pair `batch[i, j, [2k, 2k+1]]`, one of
    them is non-zero if and only if the other is non-zero.
    """
    assert t.all((self.feature_probability == self.feature_probability[:, [0]]))
    p = self.feature_probability[:, [0]]  # shape (n_inst, 1)

    feat_mag = t.rand((batch_size, self.cfg.n_inst, 2 * n_correlated_pairs), device=self.W.device)
    feat_set_seeds = t.rand((batch_size, self.cfg.n_inst, n_correlated_pairs), device=self.W.device)
    feat_set_is_present = feat_set_seeds <= p
    feat_is_present = einops.repeat(
        feat_set_is_present,
        "batch instances features -> batch instances (features pair)",
        pair=2,
    )
    return t.where(feat_is_present, feat_mag, 0.0)


def generate_anticorrelated_features(
    self: ToyModel, batch_size: int, n_anticorrelated_pairs: int
) -> Float[Tensor, "batch inst 2*n_anticorrelated_pairs"]:
    """
    Generates a batch of anti-correlated features. For each pair `batch[i, j, [2k, 2k+1]]`, each
    of them can only be non-zero if the other one is zero.
    """
    assert t.all((self.feature_probability == self.feature_probability[:, [0]]))
    p = self.feature_probability[:, [0]]  # shape (n_inst, 1)

    assert p.max().item() <= 0.5, "For anticorrelated features, must have 2p < 1"

    feat_mag = t.rand((batch_size, self.cfg.n_inst, 2 * n_anticorrelated_pairs), device=self.W.device)
    seed = t.rand((batch_size, self.cfg.n_inst, n_anticorrelated_pairs), device=self.W.device)
    mask = einops.rearrange(t.stack([seed, 1 - seed], dim=-1), "... feat pair -> ... (feat pair)") <= p
    return feat_mag * mask


def generate_uncorrelated_features(self: ToyModel, batch_size: int, n_uncorrelated: int) -> Tensor:
    """
    Generates a batch of uncorrelated features.
    """
    if n_uncorrelated == self.cfg.n_features:
        p = self.feature_probability
    else:
        assert t.all((self.feature_probability == self.feature_probability[:, [0]]))
        p = self.feature_probability[:, [0]]  # shape (n_inst, 1)

    if n_uncorrelated == self.cfg.n_features:
        p = self.feature_probability
    else:
        assert t.all((self.feature_probability == self.feature_probability[:, [0]]))
        p = self.feature_probability[:, [0]]  # shape (n_inst, 1)

    feat_mag = t.rand((batch_size, self.cfg.n_inst, n_uncorrelated), device=self.W.device)
    feat_seeds = t.rand((batch_size, self.cfg.n_inst, n_uncorrelated), device=self.W.device)
    return t.where(feat_seeds <= p, feat_mag, 0.0)

ToyModel.generate_batch = generate_batch


# %%
cfg = ToyModelConfig(n_inst=30, n_features=4, d_hidden=2, n_correlated_pairs=1, n_anticorrelated_pairs=1)

feature_probability = 10 ** -t.linspace(0.5, 1, cfg.n_inst).to(device)

model = ToyModel(cfg=cfg, device=device, feature_probability=feature_probability[:, None])

# Generate a batch of 4 features: first 2 are correlated, second 2 are anticorrelated
batch = model.generate_batch(batch_size=100_000)
corr0, corr1, anticorr0, anticorr1 = batch.unbind(dim=-1)

assert ((corr0 != 0) == (corr1 != 0)).float().mean() > 0.9999, "Correlated features should be active together"
assert ((corr0 != 0).float().mean(0) - feature_probability).abs().mean() < 0.002, (
    "Each correlated feature should be active with probability `feature_probability`"
)

assert not ((anticorr0 != 0) & (anticorr1 != 0)).any(), "Anticorrelated features should never be active together"
assert ((anticorr0 != 0).float().mean(0) - feature_probability).abs().mean() < 0.002, (
    "Each anticorrelated feature should be active with probability `feature_probability`"
)

# %%
class NeuronModel(ToyModel):
    def forward(self, features: Float[Tensor, "... inst feats"]) -> Float[Tensor, "... inst feats"]:
        activations = F.relu(
            einops.einsum(features, self.W, "... inst feats, inst d_hidden feats -> ... inst d_hidden")
        )
        out = F.relu(
            einops.einsum(activations, self.W, "... inst d_hidden, inst d_hidden feats -> ... inst feats")
            + self.b_final
        )
        return out


tests.test_neuron_model(NeuronModel)

# %%
cfg = ToyModelConfig(n_inst=7, n_features=10, d_hidden=5)

importance = 0.75 ** t.arange(1, 1 + cfg.n_features)
feature_probability = t.tensor([0.75, 0.35, 0.15, 0.1, 0.06, 0.02, 0.01])

model = NeuronModel(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None],
)
model.optimize(steps=10_000)


utils.plot_features_in_Nd(
    model.W,
    height=600,
    width=1000,
    title=f"Neuron model: {cfg.n_features=}, {cfg.d_hidden=}, I<sub>i</sub> = 0.75<sup>i</sup>",
    subplot_titles=[f"1 - S = {i:.2f}" for i in feature_probability.squeeze()],
    neuron_plot=True,
)

# %%
