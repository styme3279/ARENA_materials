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
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=32*7*7,out_features=self.hidden_dim_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim_size,out_features=self.latent_dim_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dim_size,out_features=self.hidden_dim_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim_size,out_features=32*7*7),
            Rearrange("b (c h w) -> b c h w",c=32,h=7,w=7),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=4,stride=2,padding=1)
        )

    def forward(
        self, x: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch 1 height width"]:
        """Returns the reconstruction of the input, after mapping through encoder & decoder."""
        out = self.encoder(x)
        out=self.decoder(out)
        return out


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
        self.loss = nn.MSELoss()

    def training_step(
        self, img: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, ""]:
        """
        Performs a training step on the batch of images in `img`. Returns the loss. Logs to wandb
        if enabled.
        """
        self.optimizer.zero_grad()
        out = self.model(img)

        loss = self.loss(out,img)
        loss.backward()
        self.optimizer.step()

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

            for i, (img,label) in tqdm(enumerate(self.trainloader)):
                img = img.to(device)
                img = mnist_trainset.transform(img)
                loss = self.training_step(img)
                self.step += 1

                if self.step == self.args.log_every_n_steps:
                    self.log_samples()

        if self.args.use_wandb:
            wandb.finish()

        return self.model


args = AutoencoderArgs(use_wandb=False)
trainer = AutoencoderTrainer(args)
autoencoder = trainer.train()

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


grid_latent = create_grid_of_latents(autoencoder, interpolation_range=(-3, 3))

# Map grid latent through the decoder
output = autoencoder.decoder(grid_latent)

# Visualize the output
utils.visualise_output(output, grid_latent, title="Autoencoder latent space visualization")

# %%
# Get a small dataset with 5000 points
small_dataset = Subset(trainset_mnist, indices=range(0, 5000))
imgs = trainset_mnist.transform(t.stack([img for img, label in small_dataset]).to(device))
labels = t.tensor([label for img, label in small_dataset]).to(device).int()

# Get the latent vectors for this data along first 2 dims, plus for the holdout data
latent_vectors = autoencoder.encoder(imgs)[:, :2]
holdout_latent_vectors = autoencoder.encoder(HOLDOUT_DATA)[:, :2]

# Plot the results
utils.visualise_input(latent_vectors, labels, holdout_latent_vectors, HOLDOUT_DATA)

# %%
