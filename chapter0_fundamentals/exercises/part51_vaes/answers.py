#%%

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import einops
import torch as t
import torchinfo
import wandb
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

import torch.nn.functional as F
from tqdm.auto import tqdm


# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part51_vaes"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

# Modules from previous exercises (you can swap in your own implementations if you've completed them).
from part2_cnns.solutions import BatchNorm2d, Conv2d, Linear, ReLU, Sequential
from part51_vaes import tests, utils
from fundamentals_utils import LiveImage, display_data, get_dataset
from plotly_utils import imshow

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
NUM_WORKERS = 2 if "google.colab" in sys.modules else min(8, os.cpu_count())
t.set_num_threads(min(4, t.get_num_threads()))



# %%

class Autoencoder(nn.Module):
    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        """Creates the encoder & decoder modules."""
        super().__init__()
        self.encoder = nn.Sequential([
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1),
            Rearrange("b c h w -> b (c h w)"),
            nn.ReLU(),
            nn.Linear(
                in_features=7*7*32,
                out_features=hidden_dim_size),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_dim_size,
                out_features=latent_dim_size),
        ])
        self.decoder = nn.Sequential([
            nn.Linear(
                in_features=latent_dim_size,
                out_features=hidden_dim_size),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_dim_size,
                out_features=7*7*3),
            Rearrange("b (c h w) -> b (c h w)"),
            nn.ReLU()
            ])
    def forward(
        self, x: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch 1 height width"]:
        """Returns the reconstruction of the input, after mapping through encoder & decoder."""
        raise NotImplementedError()


tests.test_autoencoder(Autoencoder)

