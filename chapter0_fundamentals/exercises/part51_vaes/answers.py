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
mnist_trainset = get_dataset("MNIST", train=True) # or "CELEB" or "CIFAR10"
dataloader = DataLoader(mnist_trainset, batch_size = 64)

# %%
for img, label in dataloader:
   img = img.to(device)
   label = label.to(device)
   img = mnist_trainset.transform(img) #transform AFTER moved to GPU, if applicable

# %%
# Don't copy this code in, just for reference.

def display_data(x: Tensor, nrows: int, title: str, return_fig: bool = False):
    """Displays a batch of data, using plotly. If `return_fig=True`, returns the figure instead
    of showing it (so a caller can update a single plot in place during training)."""
    ncols = x.shape[0] // nrows
    # Reshape into the right shape for plotting (make it 2D if image is monochrome)
    y = einops.rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=nrows).squeeze()
    # Normalize in the 0-1 range, then map to integer type
    y = y.float()
    y = (y - y.min()) / (y.max() - y.min())
    y = (y * 255).to(dtype=t.uint8)
    # Display data
    imshow(
        y,
        binary_string=(y.ndim == 2),
        height=50 * (nrows + 4),
        width=50 * (ncols + 5),
        title=f"{title}<br>single input shape = {x[0].shape}",
    )

# %%
trainset_mnist = get_dataset("MNIST")

# Display MNIST
x = next(iter(DataLoader(trainset_mnist, batch_size=25)))[0]
display_data(trainset_mnist.transform(x), nrows=5, title="MNIST data")

# %%
testset = get_dataset("MNIST", train=False)
HOLDOUT_DATA = dict()
for data, target in DataLoader(testset, batch_size=1):
    if target.item() not in HOLDOUT_DATA:
        HOLDOUT_DATA[target.item()] = data.squeeze()
        if len(HOLDOUT_DATA) == 10:
            break
HOLDOUT_DATA = testset.transform(
    t.stack([HOLDOUT_DATA[i] for i in range(10)]).to(device).unsqueeze(1)
)

display_data(HOLDOUT_DATA, nrows=1, title="MNIST holdout data")

# %%
class Autoencoder(nn.Module):
    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        """Creates the encoder & decoder modules."""
        super().__init__()
        self.hidden_dim_size = hidden_dim_size
        self.latent_dim_size = latent_dim_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=32*7*7, out_features=self.hidden_dim_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim_size, out_features=self.latent_dim_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dim_size, out_features=self.hidden_dim_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim_size, out_features=32*7*7),
            Rearrange("b (c h w) -> b c h w", c=32, h=7, w=7),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1),
        )

    def forward(
        self, x: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch 1 height width"]:
        """Returns the reconstruction of the input, after mapping through encoder & decoder."""
        return self.decoder(self.encoder(x))

tests.test_autoencoder(Autoencoder)

# %%
