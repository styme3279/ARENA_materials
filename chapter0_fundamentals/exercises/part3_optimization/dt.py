import sys
sys.path.insert(0, "/root/ARENA_materials/chapter0_fundamentals/exercises")
sys.path.insert(0, "/root/ARENA_materials")

import dataclasses
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear
import torch.nn.functional as F
from IPython.core.display import HTML
from IPython.display import display
from torch import Tensor, optim
from tqdm import tqdm
import wandb

from fundamentals_utils import IMAGENET_TRANSFORM, CIFAR10

from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from infrastructure.chapters.chapter0_fundamentals.master_0_3 import WandbResNetFinetuningArgs
import tests


WORLD_SIZE = min(t.cuda.device_count(), 3)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


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
        for r in range(world_size):
            if r != src:
                dist.send(tensor=tensor, dst=r)
    else:
        received_tensor = t.zeros(tensor.shape, device=tensor.device)
        dist.recv(tensor=received_tensor, src=src)
        tensor.copy_(received_tensor)

def reduce(tensor: Tensor, rank: int, world_size: int, dst: int = 0, op: Literal["sum", "mean"] = "sum"):
    """
    Reduces tensors to rank `dst`, so this process contains the sum or mean of all tensors across
    processes.
    """
    if rank == dst:
        for r in range(world_size):
            if r != dst:
                received_tensor = t.zeros(tensor.shape, device=tensor.device)
                dist.recv(tensor=received_tensor, src=r)
                tensor += received_tensor
        if op =="mean":
            tensor /= world_size
    else:
        dist.send(tensor=tensor, dst=dst)


def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    reduce(tensor, rank, world_size, dst=0, op=op)
    broadcast(tensor, rank, world_size, src=0)


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
        self.model = ResNet34()

        # Broadcast model weights
        for p in self.model.parameters():
            broadcast(p.data, self.rank, self.args.world_size, src=0)
    
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # Load dataset
        self.trainset = CIFAR10(train=True)
        self.testset = CIFAR10(train=False)

        self.train_sampler = t.utils.data.DistributedSampler(self.trainset, num_replicas=self.args.world_size, rank=self.rank)
        self.test_sampler = t.utils.data.DistributedSampler(self.testset, num_replicas=self.args.world_size, rank=self.rank)
        

        self.train_loader = t.utils.data.DataLoader(
            self.trainset,
            batch_size=self.args.batch_size,
            sampler=self.train_sampler
        )
        self.test_loader = t.utils.data.DataLoader(
            self.testset,
            batch_size=self.args.batch_size,
            sampler=self.train_sampler
        )

        self.logged_variables = {"loss": [], "accuracy": []}
        self.examples_seen = 0

        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_name,
            config=dataclasses.asdict(self.args),
            mode="online" if self.args.use_wandb else "disabled",
        )

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:

        # CPU -> GPU first, then perform preprocessing on GPU.
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        # Gradient synchronization
        for p in self.model.parameters():
            all_reduce(p.data, self.rank, self.args.world_size, op="sum")

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.examples_seen += imgs.shape[0]
        self.logged_variables["loss"].append(loss.item())
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct = t.zeros(size=(1,), device=self.device)

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating"):
            # CPU -> GPU first, then perform preprocessing on GPU.
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

        all_reduce(total_correct, self.rank, self.args.world_size, op="mean")

        accuracy = total_correct[0].item()
        self.logged_variables["accuracy"].append(accuracy)
        return accuracy

    def train(self):
        self.pre_training_setup()
        
        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
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

        return self.logged_variables


def dist_train_resnet_from_scratch(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size, use_wandb=False)  # flip to True to log to wandb
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":

    world_size = 2
    dist_train_resnet_from_scratch(0, world_size)
