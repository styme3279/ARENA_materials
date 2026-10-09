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
img_size = 64
img_channels = 3
hidden_channels = [128, 256, 512]


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
            nn.Linear(self.latent_dim_size, 512*8*8, bias=False),
            Rearrange("b (c w h) -> b c w h", c=512, w=8, h=8),
            nn.BatchNorm2d(512),
            nn.ReLU(.2),
        )
        self.hidden_layers = nn.Sequential(
            nn.ConvTranspose2d(512,256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            LeakyReLU(.2),
            nn.ConvTranspose2d(256,128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            LeakyReLU(.2),
            nn.ConvTranspose2d(128,3, kernel_size=4, stride=2, padding=1, bias=False),
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
            nn.Conv2d(img_channels, hidden_channels[0], kernel_size=4, stride=2, padding=1, bias=False),
            LeakyReLU(.2),
            *[layer for i in range(img_channels-1) for layer in [
                nn.Conv2d(self.hidden_channels[i], self.hidden_channels[i+1], kernel_size=4, stride=2, padding=1, bias=False), 
                nn.BatchNorm2d(self.hidden_channels[i+1]), 
                LeakyReLU(.2)]],
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*8*8, 1, bias=False),
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

print_param_count(Generator(), solutions.DCGAN().netG)
print_param_count(Discriminator(), solutions.DCGAN().netD)

# %%

model = DCGAN().to(device)
x = t.randn(3, 100).to(device)
print(torchinfo.summary(model.netG, input_data=x), end="\n\n")
print(torchinfo.summary(model.netD, input_data=model.netG(x)))

# %%
import time
for i in tqdm(range(5), position=0):
    for j in tqdm(range(4), position=1, leave=False):
        time.sleep(1)
# %%
def initialize_weights(model: nn.Module) -> None:
    """
    Initializes weights according to the DCGAN paper (details at the end of page 3 of the DCGAN
    paper), by modifying the weights of the model in place.
    """
    for module in model.modules():
        # Match torch's `nn.ConvTranspose2d` (used by default in the GAN code) AND any class
        # named `ConvTranspose2d` (so the VAEs section's bonus implementation is also picked up
        # if a student swaps it in).
        is_conv_transpose = isinstance(module, nn.ConvTranspose2d) or type(module).__name__ == "ConvTranspose2d"
        if is_conv_transpose or isinstance(module, (Conv2d, Linear)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)

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
        pred_r = self.model.netD(img_real)
        pred_f = self.model.netD(self.model.netG(img_fake))
        loss_d = -t.logsigmoid(pred_r).mean() - t.logsigmoid(-pred_f).mean()
        if self.args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.netD.parameters(), self.args.clip_grad_norm)
        loss_d.backward()
        self.optD.step()
        self.step += 1
        if self.args.use_wandb:
            wandb.log({'step':self.step, 'loss_d':loss_d})
        if self.step % self.args.log_every_n_steps == 0:
            self.log_samples()
        return loss_d

    def training_step_generator(
        self, img_fake: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, ""]:
        """
        Performs a gradient step on the generator to minimize -log(D(G(z))). Logs to wandb if enabled.
        """
        self.optG.zero_grad()
        pred_f = self.model.netD(img_fake)
        loss_g = -t.logsigmoid(pred_f).mean()
        if self.args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.netG.parameters(), self.args.clip_grad_norm)
        loss_g.backward()
        self.optG.step()
        self.step += 1
        if self.args.use_wandb:
            wandb.log({'step':self.step, 'loss_g':loss_g})
        if self.step % self.args.log_every_n_steps == 0:
            self.log_samples()
        return loss_g

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
            for img_real, label in tqdm(self.trainloader, position=1):
                
                img_real = self.trainset.transform(img_real.to(device))
                noise = t.randn(img_real.shape[0], self.args.latent_dim_size).to(device)
                img_fake = self.model.netG(noise)
            
                loss_d = self.training_step_discriminator(img_real, img_fake.detach())
                loss_g = self.training_step_generator(img_fake)


                progress_bar.set_description(f"epoch: {e}, loss_g: {loss_g}, loss_d: {loss_d}", refresh=False)
                progress_bar.update()

        progress_bar.close()
        if self.args.use_wandb:
            wandb.finish()

        return self.model

# %%
# Arguments for CelebA
args = DCGANArgs(
    dataset="CELEB",
    hidden_channels=[128, 256, 512],
    batch_size=64,  # if you get OOM errors, reduce this!
    epochs=5,
    lr_D=8e-4,  # separate LRs for generator and discriminator
    lr_G=2e-4,
    use_wandb=False,
    compile=False, # toggle me once you have the bugs ironed out
)
trainer = DCGANTrainer(args)
dcgan_celeb = trainer.train()

# %%
