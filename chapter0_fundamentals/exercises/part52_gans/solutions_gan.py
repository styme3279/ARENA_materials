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
        pos_exp = t.exp(x)
        neg_exp = t.exp(-x)
        return (pos_exp - neg_exp) / (pos_exp + neg_exp)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return t.where(x > 0, x, self.negative_slope * x)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


tests.test_Tanh(Tanh)
tests.test_LeakyReLU(LeakyReLU)

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
            nn.Linear(latent_dim_size, first_size),
            Rearrange("b (c h w) -> b c h w", h=first_height, w=first_height),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU()
        )
        hidden_layers_list = []
        for i in range(n_layers-1):
            hidden_layers_list.append(nn.ConvTranspose2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i+1], kernel_size=4, stride=2, padding=1, bias=False))
            hidden_layers_list.append(nn.BatchNorm2d(hidden_channels[i+1]))
            hidden_layers_list.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*hidden_layers_list)

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

        last_height = img_size // (2**n_layers)
        last_size = hidden_channels[-1] * (last_height**2)

        self.hidden_channels = hidden_channels
        hidden_layers_list = []
        for i in range(n_layers-1):
            hidden_layers_list.append(nn.Conv2d(hidden_channels[i], hidden_channels[i+1], kernel_size=4, stride=2, padding=1, bias=False))
            hidden_layers_list.append(nn.BatchNorm2d(hidden_channels[i+1]))
            hidden_layers_list.append(LeakyReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers_list)
        self.classifier = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(last_size, 1)
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
model = DCGAN().to(device)
x = t.randn(3, 100).to(device)
print(torchinfo.summary(model.netG, input_data=x), end="\n\n")
print(torchinfo.summary(model.netD, input_data=model.netG(x)))

# %%

def initialize_weights(model: nn.Module) -> None:
    """
    Initializes weights according to the DCGAN paper (details at the end of page 3 of the DCGAN
    paper), by modifying the weights of the model in place.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, BatchNorm2d)):
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(module.bias.data, 0)

        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, Conv2d, Linear)):
            nn.init.normal_(module.weight.data, mean=0.0, std=0.02)

tests.test_initialize_weights(initialize_weights, nn.ConvTranspose2d, Conv2d, Linear, nn.BatchNorm2d)

# %%
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
        initialize_weights(self.netD)
        initialize_weights(self.netG)

model = DCGAN().to(device)
x = t.randn(3, 100).to(device)
print(torchinfo.summary(model.netG, input_data=x), end="\n\n")
print(torchinfo.summary(model.netD, input_data=model.netG(x)))

# %%

@dataclass
class DCGANArgs:
    """
    Class for the arguments to the DCGAN (training and architecture).
    Note, we use field(default_factory=...) when our default value is a mutable object.
    """

    # architecture
    latent_dim_size: int = 100
    hidden_channels: list[int] = field(default_factory=lambda: [128, 256, 512])

    # data & training
    dataset: Literal["MNIST", "CELEB"] = "CELEB"
    batch_size: int = 64
    epochs: int = 3
    lr: float = 0.0002
    lr_G: float | None = None  # generator LR (TTUR); falls back to `lr` if None
    lr_D: float | None = None  # discriminator LR (TTUR); falls back to `lr` if None
    betas: tuple[float, float] = (0.5, 0.999)
    clip_grad_norm: float | None = 1.0

    # performance
    compile: bool = False  # optional ~2x speed-up, off by default

    # logging
    use_wandb: bool = False
    wandb_project: str | None = "day5-gan"
    wandb_name: str | None = None
    log_every_n_steps: int = 250

    def __post_init__(self):
        # Two Time-Scale Update Rule (TTUR): allow separate generator/discriminator learning
        # rates. Both default to the shared `lr` unless explicitly overridden.
        if self.lr_G is None:
            self.lr_G = self.lr
        if self.lr_D is None:
            self.lr_D = self.lr


class DCGANTrainer:
    def __init__(self, args: DCGANArgs):
        self.args = args

        self.trainset = get_dataset(self.args.dataset, transform=TANH_RANGE_TRANSFORM)
        # `drop_last=True` keeps every batch the same shape, which matters when we compile below
        self.trainloader = DataLoader(
            self.trainset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, drop_last=True
        )

        img_batch = self.trainset.transform(next(iter(self.trainloader))[0])
        batch, img_channels, img_height, img_width = img_batch.shape
        assert img_height == img_width

        self.model = DCGAN(args.latent_dim_size, img_height, img_channels, args.hidden_channels).to(device).train()

        if args.compile and device.type == "cuda":
            self.model.netG = t.compile(self.model.netG)
            self.model.netD = t.compile(self.model.netD)

        self.optG = t.optim.Adam(self.model.netG.parameters(), lr=args.lr_G, betas=args.betas)
        self.optD = t.optim.Adam(self.model.netD.parameters(), lr=args.lr_D, betas=args.betas)

        # The *same* noise every time we log samples, so successive plots show a single set of faces
        # improving rather than a fresh random set each time.
        self.fixed_noise = t.randn(10, args.latent_dim_size, device=device)

    def training_step_discriminator(
        self,
        img_real: Float[Tensor, "batch channels height width"],
        img_fake: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, ""]:
        """
        Generates a real and fake image, and performs a gradient step on the discriminator to
        minimize -(log(D(x)) + log(1-D(G(z)))). Logs to wandb if enabled.
        """
        self.optD.zero_grad()
        pred_real_img = self.model.netD(img_real)
        pred_fake_img = self.model.netD(img_fake)
        loss = -(t.log(pred_real_img) + t.log(1-pred_fake_img))


    def training_step_generator(
        self, img_fake: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, ""]:
        """
        Performs a gradient step on the generator to minimize -log(D(G(z))). Logs to wandb if enabled.
        """
        self.optG.zero_grad()
        noise = t.normal((img_fake.size[0], self.args.latent_dim_size), device=device)
        G = self.model.netG(noise)
        D = self.model.netD(G).detach()
        loss = -t.log(D)
        loss.backward()
        self.optG.step()

    @t.inference_mode()
    def log_samples(self) -> None:
        """
        Performs evaluation by passing the 10 fixed noise vectors in `self.fixed_noise` through the
        generator, then optionally logging the results to Weights & Biases.
        """
        assert self.step > 0, "First call should come after a training step. Remember to increment `self.step`."
        self.model.netG.eval()

        output = self.model.netG(self.fixed_noise)
        # Clip values to make the visualization clearer
        output = output.clamp(output.quantile(0.01), output.quantile(0.99))
        # Log to weights and biases
        if self.args.use_wandb:
            output = einops.rearrange(output, "b c h w -> b h w c").cpu().numpy()
            wandb.log({"images": [wandb.Image(arr) for arr in output]}, step=self.step)
        else:
            self.live_image.update(output)

        self.model.netG.train()

    def train(self) -> DCGAN:
        """Performs a full training run."""
        self.step = 0
        self.live_image = LiveImage()  # `log_samples` overwrites this in place
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)

        # One progress bar for the whole run (rather than a new one each epoch)
        progress_bar = tqdm(total=self.args.epochs * len(self.trainloader), ascii=True)

        for epoch in range(self.args.epochs):
            for img_real, label in self.trainloader:
                # YOUR CODE HERE - fill in the training step for generator & discriminator

        progress_bar.close()
        if self.args.use_wandb:
            wandb.finish()

        return self.model
