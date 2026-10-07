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

from chapter0_fundamentals.exercises.fundamentals_utils import CIFAR10
from chapter0_fundamentals.exercises.part2_cnns.solutions import ResNet34
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
        self.model = ResNet34
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )


        self.trainset = CIFAR10(train=True)
        self.testset = CIFAR10(train=False)

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

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        # CPU -> GPU first, then perform preprocessing on GPU.
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = IMAGENET_TRANSFORM(imgs)

        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.examples_seen += imgs.shape[0]
        self.logged_variables["loss"].append(loss.item())
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
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

        accuracy = total_correct / total_samples
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
    mp.spawn(
        run_simple_model,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
