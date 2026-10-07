# %%

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
#%%
# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


MAIN = __name__ == "__main__"
#%%
import part3_optimization.tests as tests
from fundamentals_utils import IMAGENET_TRANSFORM, CIFAR10
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

# Cap the number of CPU threads PyTorch uses. On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


WORLD_SIZE = min(t.cuda.device_count(), 3)
SRC_RANK = 0

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


# %%
assert t.cuda.is_available()
assert t.cuda.device_count() > 1, "This example requires at least 2 GPUs per machine"


# %%
def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank `src` to all other ranks.
    """
    # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")
    tensor = tensor.to(device)

    if rank == src:
        for trg in range(world_size):
            if trg != src:
                # Create a tensor, send it to rank 1
                print(f"{rank=}, {device=}, sending {tensor=}")
                dist.send(tensor, dst=trg)
    else:
        # Receive tensor from rank 0 (it needs to be on the GPU before receiving)
        print(f"{rank=}, {device=}, creating {tensor=}")
        dist.recv(tensor, src=src)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, {device=}, received {tensor=}")

# %%
def reduce(tensor, rank, world_size, dst=0, op: Literal["sum", "mean"] = "sum"):
    """
    Reduces tensors to rank `dst`, so this process contains the sum or mean of all tensors across
    processes.
    """
    device = t.device(f"cuda:{rank}")
    tensor = tensor.to(device)

    if rank == dst:
        for src in range(world_size):
            if src != dst:
            # Create a tensor, send it to rank 1
                print(f"{rank=}, {device=}, sending {tensor=}")
                temp_tensor = t.empty_like(tensor)
                dist.recv(temp_tensor, src=src)
                tensor += temp_tensor
        if op == "mean":
            tensor /= world_size
    else:
        # Receive tensor from rank 0 (it needs to be on the GPU before receiving)
        print(f"{rank=}, {device=}, creating {tensor=}")
        dist.send(tensor, dst=dst)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, {device=}, received {tensor=}")

# %%
def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """

    reduce(tensor, rank, world_size, SRC_RANK, op)
    broadcast(tensor, rank, world_size, SRC_RANK)

# %%
def get_untrained_resnet(n_classes: int) -> ResNet34:
    """
    Gets untrained resnet using code from part2_cnns.solutions (you can replace this with your
    implementation).
    """
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet

# %%
@dataclass
class ResNetFinetuningArgs:
    n_classes: int = 10
    batch_size: int = 128
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

# %%
@dataclass
class WandbResNetFinetuningArgs(ResNetFinetuningArgs):
    """Contains new params for use in wandb.init, as well as all the ResNetFinetuningArgs params."""

    wandb_project: str | None = "day3-resnet"
    wandb_name: str | None = None
    use_wandb: bool = False

# %%
@dataclass
class DistResNetTrainingArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training"


class DistResNetTrainer:
    args: DistResNetTrainingArgs

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")

    def pre_training_setup(self):
        self.model: ResNet34 = get_untrained_resnet()
        for param in self.model.parameters():
            broadcast(param.data, self.rank, self.args.world_size, SRC_RANK)
        
        if self.rank == SRC_RANK:
            self.optimizer = AdamW(
                self.model.out_layers[-1].parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )

        self.trainset = CIFAR10(train=True)
        self.testset = CIFAR10(train=False)

        self.train_sampler = t.utils.data.DistributedSampler(
            self.trainset,
            num_replicas=self.args.world_size, # we'll divide each batch up into this many random sub-batches
            rank=self.rank, # this determines which sub-batch this process gets
        )
        self.train_loader = t.utils.data.DataLoader(
            self.trainset,
            self.args.batch_size, # this is the sub-batch size, i.e. the batch size that each GPU gets
            sampler=self.train_sampler, 
            pin_memory=True,  # page-locked host memory makes the CPU -> GPU copy asynchronous and faster. For small data, may not make a difference.
        )

        self.examples_seen = 0

        if self.args.use_wandb and self.rank == SRC_RANK:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
            wandb.watch(models=[self.model], log="all", log_freq=10)

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        # CPU -> GPU first, then perform preprocessing on GPU.
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = IMAGENET_TRANSFORM(imgs)

        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        for param in self.model.parameters():
            all_reduce(param.grad, self.rank, self.args.world_size, "sum")
                

        # if self.
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.examples_seen += imgs.shape[0] * world_size

        if self.args.use_wandb:
            wandb.log({"loss": loss}, self.examples_seen)

        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()


def dist_train_resnet_from_scratch(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size, use_wandb=False)  # flip to True to log to wandb
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


if MAIN:
    world_size = t.cuda.device_count()
    mp.spawn(
        dist_train_resnet_from_scratch,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

# %%