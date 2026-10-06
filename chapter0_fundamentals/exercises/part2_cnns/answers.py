# %%
import json
import sys
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part2_cnns.tests as tests
import part2_cnns.utils as utils
from fundamentals_utils import CIFAR10
from plotly_utils import line

# On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))

# %%
class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(t.zeros((1)),x)


tests.test_relu(ReLU)

# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.weight = nn.Parameter(t.rand((out_features, in_features)), requires_grad=True)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(t.zeros((out_features)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        mul = x @ self.weight.T
        if self.bias is not None:
            return mul + self.bias
        return mul

    def extra_repr(self) -> str:
        return f"weights={self.weight.shape}, bias={self.bias.shape}, ..."

print(Linear(512,64))

tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)
# %%
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# If this is CPU, we recommend figuring out how to get cuda access (or MPS if you're on a Mac).
print(device)
# %%
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=28 * 28, out_features=100)
        self.relu = ReLU()
        self.linear2 = Linear(in_features=100, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.relu(self.linear1(self.flatten(x))))


tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)

# %%
MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(t.float32, scale=True),
        transforms.Normalize([0.1307], [0.3081]),
    ]
)


def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of MNIST training data."""

    # Get original datasets, which are downloaded to "./data" for future use
    mnist_trainset = datasets.MNIST(exercises_dir / "data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(exercises_dir / "data", train=False, download=True, transform=MNIST_TRANSFORM)

    # # Return a subset of the original datasets
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# Get the first batch of test data, by starting to iterate over `mnist_testloader`
for img_batch, label_batch in mnist_testloader:
    print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
    break

# Get the first datapoint in the test set, by starting to iterate over `mnist_testset`
for img, label in mnist_testset:
    print(f"{img.shape=}\n{label=}\n")
    break

t.testing.assert_close(img, img_batch[0])
assert label == label_batch[0].item()

# %%
model = SimpleMLP().to(device)

batch_size = 128
epochs = 3

mnist_trainset, _ = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []

for epoch in range(epochs):
    pbar = tqdm(mnist_trainloader)

    for imgs, labels in pbar:
        # Move data to device, perform forward pass
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)

        # Calculate loss, perform backward pass
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update logs & progress bar
        loss_list.append(loss.item())
        pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{loss:.3f}", refresh=False)

line(
    loss_list,
    x_max=epochs * len(mnist_trainset),
    labels={"x": "Examples seen", "y": "Cross entropy loss"},
    title="SimpleMLP training on MNIST",
    width=700,
)

# %%
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.

        We assume kernel is square, with height = width = `kernel_size`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        sf = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(sf * (2 * t.rand(out_channels, in_channels, kernel_size, kernel_size) - 1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d, which you can import."""
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


tests.test_conv2d_module(Conv2d)
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")


# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a
        `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right
        branch. Declare it second using another `Sequential`.
        """
        super().__init__()
        is_shape_preserving = (first_stride == 1) and (in_feats == out_feats)  # determines if right branch is identity
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_feats)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_feats)
        self.conv = nn.Conv2d(in_channels=out_feats, out_channels=out_feats, kernel_size=3, stride=1, padding=1)
        self.strided_conv = nn.Conv2d(in_channels=in_feats, out_channels=out_feats, stride=first_stride, kernel_size=3, padding=1)
        # optional part
        self.optional_strided_conv = None
        self.optional_batch_norm = None
        if not is_shape_preserving:
            self.optional_strided_conv = nn.Conv2d(kernel_size=1, in_channels=in_feats, out_channels=out_feats, stride=first_stride, padding=0)
            self.optional_batch_norm = nn.BatchNorm2d(num_features=out_feats)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass. If no downsampling block is present, the addition should just add
        the left branch's output to the input.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)
        """
        y1 = self.strided_conv(x)
        y1 = self.batch_norm1(y1)
        y1 = self.relu1(y1)
        y1 = self.conv(y1)
        y1 = self.batch_norm2(y1)

        y2 = x
        if self.optional_strided_conv is not None:
            y2 = self.optional_strided_conv(x)
            y2 = self.optional_batch_norm(y2)

        y = y1 + y2
        return self.relu2(y)

tests.test_residual_block(ResidualBlock)

# %%
from collections import OrderedDict
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """
        An n_blocks-long sequence of ResidualBlock where only the first block uses the provided
        stride.
        """
        super().__init__()
        # blocks = [("0", ResidualBlock(in_feats=in_feats, out_feats=out_feats, first_stride=first_stride))]
        # for i in range(n_blocks):
        #     if i == 0:
        #         blocks.append((str(i), ResidualBlock(in_feats=in_feats, out_feats=out_feats, first_stride=first_stride)))
        #     else: 
        #         blocks.append((str(i), ResidualBlock(in_feats=in_feats, out_feats=out_feats)))
        # self.blocks = nn.Sequential(OrderedDict(blocks))
        # print(self.blocks)
        self.blocks = nn.Sequential(
            ResidualBlock(in_feats, out_feats, first_stride),
            *[ResidualBlock(out_feats, out_feats) for _ in range(n_blocks - 1)],
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.blocks(x)

tests.test_block_group(BlockGroup)
# %%
class AveragePool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return t.mean(x, dim=(2, 3))

# %%
class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, " num_features"]
    running_var: Float[Tensor, " num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize each channel.

        In training mode, normalize using the biased variance `x.var(..., correction=0)`, but update
        `running_var` using the unbiased variance `x.var(..., correction=1)`.
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        # Calculating mean and var over all dims except for the channel dim
        if self.training:
            # Take mean over all dimensions except the feature dimension
            mean = x.mean(dim=(0, 2, 3))
            # Normalize with the batch's actual variance (biased, dividing by n)
            var = x.var(dim=(0, 2, 3), correction=0)
            # Updating running mean and variance, in line with PyTorch documentation.
            # Disable gradients as they are not parameters
            with t.no_grad():
                # running_var estimates the population variance, so use the unbiased estimator (dividing by n-1)
                var_unbiased = x.var(dim=(0, 2, 3), correction=1)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_unbiased
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        # Rearranging these so they can be broadcasted
        reshape = lambda x: einops.rearrange(x, "channels -> 1 channels 1 1")

        # Normalize, then apply affine transformation from self.weight & self.bias
        x_normed = (x - reshape(mean)) / (reshape(var) + self.eps).sqrt()
        x_affine = x_normed * reshape(self.weight) + reshape(self.bias)
        return x_affine

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])

class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        out_feats0 = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        self.conv = nn.Conv2d(3, out_feats0, kernel_size=7, stride=2, padding=3)
        self.batch_norm = BatchNorm2d(out_feats0)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block_groups = []
        for i in range(len(n_blocks_per_group)):
            block_group = BlockGroup(
                n_blocks = self.n_blocks_per_group[i],
                in_feats=[64, *self.out_features_per_group][i],
                out_feats=self.out_features_per_group[i],
                first_stride=self.first_strides_per_group[i]
            )
            block_groups.append(block_group)
        self.block_groups = nn.Sequential(*block_groups)

        self.avg_pool = AveragePool()
        self.linear = nn.Linear(in_features=self.out_features_per_group[-1], out_features=n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        """
        y = self.conv(x)
        y = self.batch_norm(y)
        y = self.relu(y)
        y = self.max_pool(y)
        y = self.block_groups(y)
        y = self.avg_pool(y)
        return self.linear(y)

my_resnet = ResNet34()

# (1) Test via helper function `print_param_count`
target_resnet = models.resnet34()  # without supplying a `weights` argument, we just initialize with random weights
#utils.print_param_count(my_resnet, target_resnet)

# (2) Test via `torchinfo.summary`
print("My model:", torchinfo.summary(my_resnet, input_size=(1, 3, 64, 64)), sep="\n")
print(
    "\nReference model:",
    torchinfo.summary(target_resnet, input_size=(1, 3, 64, 64), depth=2),
    sep="\n",
)

# %%
