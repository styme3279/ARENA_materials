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
        x = t.clamp(x,-80,80)
        return ((t.exp(x)) - t.exp(-x)) / (t.exp(x)+t.exp(-x))

class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return t.where(x>0,x,self.negative_slope*x)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


tests.test_Tanh(Tanh)
tests.test_LeakyReLU(LeakyReLU)

# %%
img_size = 64
img_channels = 3
hidden_channels = [128, 256, 512]
# %%
img_size = 64
img_channels = 3
hidden_channels = [128, 256, 512]
latent_dim_size = 100

# %%

class Generator(nn.Module):
    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        """
        Implements the generator architecture from the DCGAN paper (the diagram at the top
        of page 4). We assume the size of the activations doubles at each layer (so image
        size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            latent_dim_size:
                the size of the latent dimension, i.e. the input to the generator
            img_size:
                the size of the image, i.e. the output of the generator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the generator (starting closest
                to the image and going inward, i.e. in reverse-chronological order for the
                generator, which is why we reverse this list below)
        """
        n_layers = len(hidden_channels)
        assert img_size % (2**n_layers) == 0, (
            "activation size must double at each layer"
        )

        super().__init__()
        # Reverse hidden channels, so they're in chronological order
        hidden_channels = hidden_channels[::-1]
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels

        # Define the first layer, i.e. latent dim -> (512, 8, 8) and reshape
        first_height = img_size // (2**n_layers)
        first_size = hidden_channels[0] * (first_height**2)
        
        
        
        self.project_and_reshape = nn.Sequential(
            nn.Linear(self.latent_dim_size,8*8*512, bias=False),
            Rearrange('b (c h w) -> b c h w',c=512,h=8,w=8),
            BatchNorm2d(512),
            nn.ReLU()
        )
        self.hidden_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False),
            BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
            BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=3,kernel_size=4,stride=2,padding=1,bias=False)
        )

    def forward(
        self, x: Float[Tensor, "batch latent"]
    ) -> Float[Tensor, "batch channels height width"]:
        x = self.project_and_reshape(x)
        x = self.hidden_layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        """
        Implements the discriminator architecture from the DCGAN paper (the mirror image of
        the diagram at the top of page 4). We assume the size of the activations halves at
        each layer (so image size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            img_size:
                the size of the image, i.e. the input of the discriminator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the discriminator (starting
                closest to the image and going inward, i.e. in chronological order for the
                discriminator)
        """
        n_layers = len(hidden_channels)
        assert img_size % (2**n_layers) == 0, "activation size must halve at each layer"

        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        
        
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.img_channels,out_channels=self.hidden_channels[0],kernel_size=4,stride=2,padding=1),
            BatchNorm2d(self.hidden_channels[0]),
            nn.LeakyReLU(),
            *(
                layer
                for i in range(len(self.hidden_channels)-1)
                for layer in [
                nn.Conv2d(in_channels=self.hidden_channels[i],out_channels=self.hidden_channels[i+1],kernel_size=4,stride=2,padding=1),
                BatchNorm2d(self.hidden_channels[i+1]),
                nn.LeakyReLU(0.2)
            ]
            )
        )
        self.classifier = nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(512*8*8,1, bias=False)
        )

    def forward(
        self, x: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch"]:
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x.squeeze(-1)  # remove dummy `out_channels` dimension


class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.netD = Discriminator(img_size, img_channels, hidden_channels)
        self.netG = Generator(latent_dim_size, img_size, img_channels, hidden_channels)

# %%
from part2_cnns.utils import print_param_count
from part52_gans import solutions

#print_param_count(Generator(), solutions.DCGAN().netG)
print_param_count(Discriminator(), solutions.DCGAN().netD)
# %%
