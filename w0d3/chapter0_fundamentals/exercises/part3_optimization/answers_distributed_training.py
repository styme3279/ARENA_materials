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

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


MAIN = __name__ == "__main__"

import part3_optimization.tests as tests
from fundamentals_utils import IMAGENET_TRANSFORM, CIFAR10
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

# Cap the number of CPU threads PyTorch uses. On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
# %%
WORLD_SIZE = min(t.cuda.device_count(), 3)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


def send_receive(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Send tensor to rank 1
        sending_tensor = t.zeros(1)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)
    elif rank == 1:
        # Receive tensor from rank 0
        received_tensor = t.ones(1)
        print(f"{rank=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, received {received_tensor=}")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2  # simulate 2 processes
    mp.spawn(
        send_receive,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
# %%
def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank `src` to all other ranks.
    """
    if rank == src:
        # Send tensor to rank 1
        #print(f"{rank}, sending {tensor}")
        for rank in range(world_size):
            if rank != src:
                dist.send(tensor=tensor, dst=rank)
    else:
        # Receive tensor from rank 0
        dist.recv(tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        #print(f"{rank}, received {tensor}")


if __name__ == "__main__":
    tests.test_broadcast(broadcast, WORLD_SIZE)
# %%
def reduce(tensor, rank, world_size, dst=0, op: Literal["sum", "mean"] = "sum"):
    """
    Reduces tensors to rank `dst`, so this process contains the sum or mean of all tensors across
    processes.
    """
    if rank != dst:
        #print(tensor, dst)
        dist.send(tensor, dst=dst)
        #print(f"Sending tensor from {rank=} to {dst=}")
    else:
        tensors = [tensor]
        #print(f"Before receive: {tensors}")
        for rank in range(world_size):
            #print(f"Waiting for {rank=}")
            if rank != dst:
                received_tensor = t.zeros_like(tensor)
                dist.recv(received_tensor, src=rank)
                tensors.append(received_tensor)
                #print(f"Received tensor from {rank=} on {dst=}")
        if op == "sum":
            #print(f"Sum: {t.stack(tensors)}")
            #print(f"Sum: {t.stack(tensors).sum(dim=0)}")
            tensor.copy_(t.stack(tensors).sum(dim=0))
        elif op == "mean":
            #print(f"Mean: {t.stack(tensors)}")
            #print(f"Mean: {t.stack(tensors).sum(dim=0)}")
            tensor.copy_(t.stack(tensors).mean(dim=0))

def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    reduce(tensor, rank, world_size, dst=0, op=op)
    broadcast(tensor, rank, world_size, src=0)

if MAIN:
    tests.test_reduce(reduce, WORLD_SIZE)
    tests.test_all_reduce(all_reduce, WORLD_SIZE)

# %%
class SimpleModel(t.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.param = t.nn.Parameter(t.tensor([2.0]))

    def forward(self, x: Tensor):
        return x - self.param


def run_simple_model(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")
    model = SimpleModel().to(device)  # Move the model to the device corresponding to this process
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)

    input = t.tensor([rank], dtype=t.float32, device=device)
    output = model(input)
    loss = output.pow(2).sum()
    loss.backward()  # Each rank has separate gradients at this point

    print(f"Rank {rank}, before all_reduce, grads: {model.param.grad=}")
    all_reduce(model.param.grad, rank, world_size)  # Synchronize gradients
    print(f"Rank {rank}, after all_reduce, synced grads (summed over processes): {model.param.grad=}")

    optimizer.step()  # Step with the optimizer (this will update all models the same way)
    print(f"Rank {rank}, new param: {model.param.data}")

    dist.destroy_process_group()


if MAIN:
    world_size = 2
    mp.spawn(
        run_simple_model,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

# %%
def get_untrained_resnet(n_classes: int) -> ResNet34:
    """
    Gets untrained resnet using code from part2_cnns.solutions (you can replace this with your
    implementation).
    """
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet

@dataclass
class ResNetFinetuningArgs:
    n_classes: int = 10
    batch_size: int = 128
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class WandbResNetFinetuningArgs(ResNetFinetuningArgs):
    """Contains new params for use in wandb.init, as well as all the ResNetFinetuningArgs params."""

    wandb_project: str | None = "day3-resnet"
    wandb_name: str | None = None
    use_wandb: bool = False

@dataclass
class DistResNetTrainingArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training"


class DistResNetTrainer:
    args: DistResNetTrainingArgs
    examples_seen: int = 0  # tracking examples seen (used as step for wandb)

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")

    def pre_training_setup(self):
        """Initializes the wandb run using `wandb.init` and `wandb.watch`."""
        self.model = get_untrained_resnet(self.args.n_classes).to(device)
        self.optimizer = t.optim.AdamW(
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

        self.test_sampler = t.utils.data.DistributedSampler(
            self.testset,
            num_replicas=self.args.world_size, # we'll divide each batch up into this many random sub-batches
            rank=self.rank, # this determines which sub-batch this process gets
        )

        self.test_loader = t.utils.data.DataLoader(
            self.testset,
            self.args.batch_size, # this is the sub-batch size, i.e. the batch size that each GPU gets
            sampler=self.test_sampler, 
            pin_memory=True,  # page-locked host memory makes the CPU -> GPU copy asynchronous and faster. For small data, may not make a difference.
        )
        self.examples_seen = 0
        if self.rank == 0:
            wandb.init()

        #src_params = []
        for param in self.model.parameters():
            broadcast(param.data, self.rank, self.args.world_size)
        #    src_params.append(param) # local param contains param from src
        #self.model.parameters().data = src_params # set params from src after broadcast


    def training_step(
        self,
        imgs: Float[Tensor, "batch channels height width"],
        labels: Int[Tensor, " batch"],
    ) -> Float[Tensor, ""]:
        """Perform a gradient update step on a single batch of data."""

        # CPU -> GPU first, then perform preprocessing on GPU.
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = IMAGENET_TRANSFORM(imgs)

        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        for param in self.model.parameters():
            all_reduce(param.grad, self.rank, self.args.world_size, op="mean")
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.examples_seen += imgs.shape[0] * self.args.world_size
        if self.rank == 0:
            wandb.log({"loss": loss.item()}, self.examples_seen)
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """Evaluate the model on the test set and return the accuracy."""
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating"):
            # CPU -> GPU first, then perform preprocessing on GPU.
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs = IMAGENET_TRANSFORM(imgs)

            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        tensor = t.tensor([total_correct, total_samples], device=self.device)
        all_reduce(tensor, self.rank, self.args.world_size, op="sum")    
        total_correct, total_samples = tensor.tolist()
        accuracy = total_correct / total_samples
        
        if self.rank == 0:
            wandb.log({"accuracy": accuracy}, self.examples_seen)
        return accuracy

    def train(self) -> dict[str, list[float]]:
        self.pre_training_setup()

        accuracy = self.evaluate()

        for _ in range(self.args.epochs):
            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training")
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    ex_seen=f"{self.examples_seen:06}",
                    refresh=False,
                )

            accuracy = self.evaluate()
            pbar.set_postfix(
                loss=f"{loss:.3f}",
                accuracy=f"{accuracy:.2f}",
                ex_seen=f"{self.examples_seen:06}",
            )
        if self.rank == 0:
            wandb.finish()
        return self.logged_variables


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
