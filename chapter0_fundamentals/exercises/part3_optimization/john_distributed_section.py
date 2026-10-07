# %%
import os
from dataclasses import dataclass

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from IPython.core.display import HTML
from IPython.display import display
from torch import Tensor, optim
import einops
from typing import Literal
from part2_cnns.solutions import Linear, ResNet34
import wandb
from fundamentals_utils import IMAGENET_TRANSFORM, CIFAR10
from torch.utils.data import DataLoader, DistributedSampler

import tests


WORLD_SIZE = min(t.cuda.device_count(), 3)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

@dataclass
class ResNetFinetuningArgs:
    n_classes: int = 10
    batch_size: int = 128
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


class WandbResNetFinetuningArgs(ResNetFinetuningArgs):
    """Contains new params for use in wandb.init, as well as all the ResNetFinetuningArgs params."""

    wandb_project: str | None = "day3-resnet"
    wandb_name: str | None = None
    use_wandb: bool = False


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

def send_receive_nccl(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")

    if rank == 0:
        # Create a tensor, send it to rank 1
        sending_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, sending {sending_tensor=}")
        dist.send(sending_tensor, dst=1)
    elif rank == 1:
        # Receive tensor from rank 0 (it needs to be on the GPU before receiving)
        received_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, {device=}, received {received_tensor=}")

    dist.destroy_process_group()

def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank `src` to all other ranks.
    """
    if rank == src:
        for other_rank in range(world_size):
            if other_rank != src:
                dist.send(tensor=tensor, dst=other_rank)

    else:
        received_tensor = t.zeros_like(tensor)
        dist.recv(tensor=received_tensor, src=src)
        tensor.copy_(received_tensor)
        
def reduce(tensor, rank, world_size, dst=0, op: Literal["sum", "mean"] = "sum"):
    """
    Reduces tensors to rank `dst`, so this process contains the sum or mean of all tensors across
    processes.
    """
    if rank != dst:
        dist.send(tensor, dst=dst)
    else:
        for other_rank in range(world_size):
            if other_rank != dst:
                received_tensor = t.zeros_like(tensor)
                dist.recv(received_tensor, src=other_rank)
                tensor += received_tensor
    if op == "mean":
        tensor /= world_size


def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    reduce(tensor, rank, world_size, dst=0, op=op)
    broadcast(tensor, rank, world_size, src=0)

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

def get_untrained_resnet(n_classes: int) -> ResNet34:
    """
    Gets untrained resnet using code from part2_cnns.solutions (you can replace this with your
    implementation).
    """
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet


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
        self.model = get_untrained_resnet(self.args.n_classes).to(self.device)
        if self.args.world_size > 1:
            for param in self.model.parameters():
                broadcast(param.data, self.rank, self.args.world_size, src=0)

        self.optimizer = t.optim.AdamW(
            self.model.out_layers[-1].parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        self.trainset = CIFAR10(train=True)
        self.testset = CIFAR10(train=False)
        self.train_sampler
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        self.logged_variables = {"loss": [], "accuracy": []}
        self.examples_seen = 0
        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_name,
            config=self.args,
            mode="online" if self.args.use_wandb else "disabled",          
        )
        wandb.watch(self.model.out_layers[-1], log="all", log_freq=50)
        self.examples_seen = 0


    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError()

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


if __name__ == "__main__":
    world_size = t.cuda.device_count()
    mp.spawn(
        dist_train_resnet_from_scratch,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )


"""
if __name__ == "__main__":
    world_size = 2
    mp.spawn(
        run_simple_model,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
"""

"""
if __name__ == "__main__":
    world_size = 2  # simulate 2 processes
    mp.spawn(
        send_receive_nccl,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
    tests.test_broadcast(broadcast, WORLD_SIZE)
    tests.test_reduce(reduce, WORLD_SIZE)
    tests.test_all_reduce(all_reduce, WORLD_SIZE)
"""
# %%
