# %%
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
section = "part52_gans"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

# Modules from previous exercises (you can swap in your own implementations if you've completed them).
from part2_cnns.solutions import BatchNorm2d, Conv2d, Linear, ReLU, Sequential
from part52_gans import tests
from fundamentals_utils import TANH_RANGE_TRANSFORM, LiveImage, display_data, get_dataset
from plotly_utils import imshow

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
NUM_WORKERS = 2 if "google.colab" in sys.modules else min(8, os.cpu_count())
t.set_num_threads(min(4, t.get_num_threads()))

# %%
trainset_celeb = get_dataset("CELEB")
x = next(iter(DataLoader(trainset_celeb, batch_size=25)))[0]
display_data(trainset_celeb.transform(x), nrows=5, title="CelebA data")

# %%
class Tanh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = t.clamp(x, -80, 80)
        return (t.exp(x) - t.exp(-x)) / (t.exp(x) + t.exp(-x)) 

class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(x, t.tensor(0)) + t.minimum(self.negative_slope*x, t.tensor(0))

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


tests.test_Tanh(Tanh)
tests.test_LeakyReLU(LeakyReLU)

# %%