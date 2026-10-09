# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter0_fundamentals"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
try:
    import torchinfo
except:
    %pip install torchinfo jaxtyping einops datasets
    %pip install -U datasets

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


# Modules from previous exercises (you can swap in your own implementations if you've completed them).
from part2_cnns.solutions import BatchNorm2d, Conv2d, Linear, ReLU, Sequential
from part51_vaes import tests, utils
from fundamentals_utils import LiveImage, display_data, get_dataset
from plotly_utils import imshow

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
NUM_WORKERS = 2 if "google.colab" in sys.modules else min(8, os.cpu_count())
t.set_num_threads(min(4, t.get_num_threads()))

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
        self.encoder = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            ReLU(),
            nn.Flatten(),
            Linear(in_features=32*7*7, out_features=hidden_dim_size),
            ReLU(),
            Linear(in_features=hidden_dim_size, out_features=latent_dim_size)
        )
        self.decoder = Sequential(
            Linear(in_features=latent_dim_size, out_features=hidden_dim_size),
            ReLU(),
            Linear(in_features=hidden_dim_size, out_features=32*7*7),
            Rearrange("b (c h w) -> b c h w", c=32, h = 7, w = 7),
            ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=False),
            ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        )

    def forward(
        self, x: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch 1 height width"]:
        """Returns the reconstruction of the input, after mapping through encoder & decoder."""
        return self.decoder(self.encoder(x))


tests.test_autoencoder(Autoencoder)

# %%
@dataclass
class AutoencoderArgs:
    # architecture
    latent_dim_size: int = 5
    hidden_dim_size: int = 128

    # data / training
    dataset: Literal["MNIST", "CELEB"] = "MNIST"
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-3
    betas: tuple[float, float] = (0.5, 0.999)

    # logging
    use_wandb: bool = True
    wandb_project: str | None = "day5-autoencoder"
    wandb_name: str | None = None
    log_every_n_steps: int = 250


class AutoencoderTrainer:
    def __init__(self, args: AutoencoderArgs):
        self.args = args
        self.trainset = get_dataset(args.dataset)
        self.trainloader = DataLoader(
            self.trainset, batch_size=args.batch_size, shuffle=True
        )
        self.model = Autoencoder(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)
        self.optimizer = t.optim.Adam(
            self.model.parameters(), lr=args.lr, betas=args.betas
        )
        self.mse = nn.MSELoss()

    def training_step(
        self, img: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, ""]:
        """
        Performs a training step on the batch of images in `img`. Returns the loss. Logs to wandb
        if enabled.
        """

        img = self.trainset.transform(img.to(device))
        created_pic = self.model(img)

        loss = self.mse.forward(input=created_pic, target=img)
        loss.backward()

        self.step += 1
        if self.args.use_wandb:
            wandb.log({"loss": loss.item()}, step=self.step)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    @t.inference_mode()
    def log_samples(self) -> None:
        """
        Evaluates model on holdout data, either logging to weights & biases or displaying output.
        """
        assert self.step > 0, "First call should come after a training step. Remember to increment `self.step`."
        output = self.model(HOLDOUT_DATA)
        if self.args.use_wandb:
            output = (output - output.min()) / (output.max() - output.min())  # Normalize to [0, 1]
            output = (output * 255).to(dtype=t.uint8)  # Convert to uint8 for logging
            wandb.log({"images": [wandb.Image(arr) for arr in output.cpu().numpy()]}, step=self.step)
        else:
            self.live_image.update(t.concat([HOLDOUT_DATA, output]), nrows=2)  # top: input, bottom: reconstruction

    def train(self) -> Autoencoder:
        """Performs a full training run."""
        self.step = 0
        self.live_image = LiveImage()  # `log_samples` overwrites this in place
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            wandb.watch(self.model)

        # YOUR CODE HERE - iterate over epochs, and train your model

        for epoch in range(self.args.epochs):
            self.model.train()
            pbar = tqdm(self.trainloader)
            for img, _ in pbar:
                loss = self.training_step(img)
                pbar.set_postfix(loss=f"{loss:.3f}", step=f"{self.step=:06}", refresh=False)

                if self.step % args.log_every_n_steps == 1:
                    self.log_samples()

        if self.args.use_wandb:
            wandb.finish()

        return self.model


# args = AutoencoderArgs(use_wandb=True, wandb_project="day5-autoencoder", wandb_name="autoencoder")
# trainer = AutoencoderTrainer(args)
# autoencoder = trainer.train()

# %%

def create_grid_of_latents(
    model: nn.Module,
    interpolation_range: tuple[float, float] = (-1, 1),
    n_points: int = 11,
    dims: tuple[int, int] = (0, 1),
) -> Float[Tensor, "rows_x_cols latent_dims"]:
    """Create a tensor of zeros which varies along the 2 specified dimensions of the latent space."""
    grid_latent = t.zeros(n_points, n_points, model.latent_dim_size, device=device)
    x = t.linspace(*interpolation_range, n_points)
    grid_latent[..., dims[0]] = x.unsqueeze(-1)  # rows vary over dim=0
    grid_latent[..., dims[1]] = x  # cols vary over dim=1
    return grid_latent.flatten(0, 1)  # flatten over (rows, cols) into a single batch dimension


# grid_latent = create_grid_of_latents(autoencoder, interpolation_range=(-3, 3))

# # Map grid latent through the decoder
# output = autoencoder.decoder(grid_latent)

# # Visualize the output
# utils.visualise_output(output, grid_latent, title="Autoencoder latent space visualization")

#  # %%
# # Get a small dataset with 5000 points
# small_dataset = Subset(trainset_mnist, indices=range(0, 5000))
# imgs = trainset_mnist.transform(t.stack([img for img, label in small_dataset]).to(device))
# labels = t.tensor([label for img, label in small_dataset]).to(device).int()

# # Get the latent vectors for this data along first 2 dims, plus for the holdout data
# latent_vectors = autoencoder.encoder(imgs)[:, :2]
# holdout_latent_vectors = autoencoder.encoder(HOLDOUT_DATA)[:, :2]

# # Plot the results
# utils.visualise_input(latent_vectors, labels, holdout_latent_vectors, HOLDOUT_DATA)


 # %%
class VAE(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        super().__init__()
        self.hidden_dim_size = hidden_dim_size
        self.latent_dim_size = latent_dim_size
        self.encoder = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            ReLU(),
            nn.Flatten(),
            Linear(in_features=32*7*7, out_features=hidden_dim_size),
            ReLU(),
            Linear(in_features=hidden_dim_size, out_features=2*latent_dim_size),
            Rearrange("b (l1 l2) -> l2 b l1", l1 = 5, l2 = 2)
        )
        self.decoder = Sequential(
            Linear(in_features=latent_dim_size, out_features=hidden_dim_size),
            ReLU(),
            Linear(in_features=hidden_dim_size, out_features=32*7*7),
            Rearrange("b (c h w) -> b c h w", c=32, h = 7, w = 7),
            ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=False),
            ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        )


    def sample_latent_vector(
        self, x: Float[Tensor, "batch 1 height width"]
    ) -> tuple[
        Float[Tensor, "batch latent"],
        Float[Tensor, "batch latent"],
        Float[Tensor, "batch latent"],
    ]:
        """
        Passes `x` through the encoder, returns tuple of (sampled latent vector, mean, log std dev).
        This function can be used in `forward`, but also used on its own to generate samples for
        evaluation.
        """
        latent = self.encoder(x)
        mu = latent[0]
        logsigma = latent[1]

        sampled_vector = t.randn_like(logsigma)

        # out = mu + sigma * sampled_vector
        sigma = t.exp(logsigma)
        variance = einops.einsum(sigma, sampled_vector, '... i j , ... i j -> ... i j')
        out = mu + variance
        

        return out, mu, logsigma

        

    def forward(
        self, x: Float[Tensor, "batch 1 height width"]
    ) -> tuple[
        Float[Tensor, "batch 1 height width"],
        Float[Tensor, "batch latent"],
        Float[Tensor, "batch latent"],
    ]:
        """
        Passes `x` through the encoder and decoder. Returns the reconstructed input, as well as mu
        and logsigma.
        """
        sampled, mu, logsigma = self.sample_latent_vector(x)
        out = self.decoder(sampled)

        return out, mu, logsigma

tests.test_vae(VAE)


# %%
@dataclass
class VAEArgs(AutoencoderArgs):
    wandb_project: str | None = "day5-vae-mnist"
    beta_kl: float = 0.1


class VAETrainer:
    def __init__(self, args: VAEArgs):
        self.args = args
        self.trainset = get_dataset(args.dataset)
        self.trainloader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.model = VAE(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas)
        
        self.mse = nn.MSELoss()

    def training_step(
        self, img: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, ""]:
        """ 
        Performs a training step on the batch of images in `img`. Returns the loss. Logs to wandb
        if enabled.
        """

        img = self.trainset.transform(img.to(device))
        created_pic, mu, logsigma = self.model(img)

        loss_mse = self.mse.forward(input=created_pic, target=img)

        sigma = t.exp(logsigma)

        loss_kl_vec = (sigma**2 + mu**2 - 1) / 2 - logsigma
        # print(f"loss_kl shape: {loss_kl_vec.shape}")
        loss_kl_scalar = loss_kl_vec.sum()

        loss = loss_mse + self.args.beta_kl * loss_kl_scalar

        loss.backward()

        self.step += 1
        if self.args.use_wandb:
            wandb.log({"loss_mse": loss_mse.item()}, step=self.step)
            wandb.log({"loss_kl_scalar": loss_kl_scalar.item()}, step=self.step)
            wandb.log({"loss": loss.item()}, step=self.step)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    @t.inference_mode()
    def log_samples(self) -> None:
        """
        Evaluates model on holdout data, either logging to wandb or displaying output inline.
        """
        assert self.step > 0, "First call should come after a training step. Remember to increment `self.step`."
        output = self.model(HOLDOUT_DATA)[0]
        if self.args.use_wandb:
            output = (output - output.min()) / (output.max() - output.min())  # Normalize to [0, 1]
            output = (output * 255).to(dtype=t.uint8)  # Convert to uint8 for logging
            wandb.log({"images": [wandb.Image(arr) for arr in output.cpu().numpy()]}, step=self.step)
        else:
            self.live_image.update(t.concat([HOLDOUT_DATA, output]), nrows=2)  # top: input, bottom: reconstruction


    def train(self) -> Autoencoder:
        """Performs a full training run."""
        self.step = 0
        self.live_image = LiveImage()  # `log_samples` overwrites this in place
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            wandb.watch(self.model)

        for epoch in range(self.args.epochs):
            self.model.train()
            pbar = tqdm(self.trainloader)
            for img, _ in pbar:
                loss = self.training_step(img)
                pbar.set_postfix(loss=f"{loss:.3f}", step=f"{self.step=:06}", refresh=False)

                if self.step % args.log_every_n_steps == 1:
                    self.log_samples()

        if self.args.use_wandb:
            wandb.finish()

        return self.model

sweep_config = dict(
    method="random",
    metric=dict(name="accuracy", goal="maximize"),
    parameters=dict(
        learning_rate=dict(min=1e-4, max=1e-1, distribution="log_uniform_values"),
        batch_size=dict(values=[32, 64, 128, 256]),
        weight_decay=dict(min=1e-4, max=1e-2, distribution="log_uniform_values"),
        weight_decay_bool=dict(values=[True, False]),
    ),
)


args = VAEArgs(latent_dim_size=5, hidden_dim_size=100, use_wandb=True)
trainer = VAETrainer(args)
vae = trainer.train()



# %%


