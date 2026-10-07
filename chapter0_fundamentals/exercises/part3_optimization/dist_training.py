"""
[0.3] section 3️⃣ Distributed Training, with the solutions filled in.

The definitions follow the notebook (0.3_Optimization_exercises.ipynb) in order, under the same
headings. The `__main__` block at the bottom runs the exercises in the same order, one per line -
comment out any line to skip that step.

Run from this directory, on a machine with >= 2 GPUs:
    python dist_training.py

mp.spawn starts each child process by re-importing this file, where `__name__ != "__main__"`, so the
children only get the definitions and don't run the `__main__` block.
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

exercises_dir = Path(__file__).resolve().parents[1]
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part3_optimization.tests as tests  # noqa: E402
from fundamentals_utils import CIFAR10, IMAGENET_TRANSFORM  # noqa: E402
from part2_cnns.solutions import Linear, ResNet34  # noqa: E402


# From section 2️⃣ (Weights and Biases) - the args that DistResNetTrainingArgs builds on
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


# ===== Basic send & receiving =====

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


# ===== Collective communication primitives =====
# ===== Exercise - implement `broadcast` =====


def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank `src` to all other ranks.
    """
    if rank == src:
        for other_rank in range(world_size):
            if other_rank != src:
                dist.send(tensor, dst=other_rank)
    else:
        received_tensor = t.zeros_like(tensor)
        dist.recv(received_tensor, src=src)
        tensor.copy_(received_tensor)


# ===== Exercise - implement `all_reduce` =====


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


# ===== Full training loop =====
# ===== Exercise - complete `DistResNetTrainer` =====


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
                # dist.broadcast(param.data, src=0)

        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )

        self.trainset = CIFAR10(train=True)
        self.testset = CIFAR10(train=False)
        self.train_sampler = self.test_sampler = None
        if self.args.world_size > 1:
            self.train_sampler = DistributedSampler(self.trainset, num_replicas=self.args.world_size, rank=self.rank)
            self.test_sampler = DistributedSampler(self.testset, num_replicas=self.args.world_size, rank=self.rank)
        self.train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, sampler=self.train_sampler)
        self.test_loader = DataLoader(self.testset, batch_size=self.args.batch_size, sampler=self.test_sampler)
        self.examples_seen = 0

        if self.rank == 0:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_name,
                config=self.args,
                mode="online" if self.args.use_wandb else "disabled",
            )

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        t0 = time.time()

        # Forward pass
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        imgs = IMAGENET_TRANSFORM(imgs)
        logits = self.model(imgs)
        t1 = time.time()

        # Backward pass
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        t2 = time.time()

        # Gradient synchronization
        if self.args.world_size > 1:
            for param in self.model.parameters():
                all_reduce(param.grad, self.rank, self.args.world_size, op="mean")
                # dist.all_reduce(param.grad, op=dist.ReduceOp.SUM); param.grad /= self.args.world_size
        t3 = time.time()

        # Optimizer step, update examples seen & log data
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.examples_seen += imgs.shape[0] * self.args.world_size
        if self.rank == 0:
            wandb.log(
                {
                    "loss": loss.item(),
                    "fwd_time": (t1 - t0),
                    "bwd_time": (t2 - t1),
                    "dist_time": (t3 - t2),
                },
                step=self.examples_seen,
            )
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating", disable=self.rank != 0):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            imgs = IMAGENET_TRANSFORM(imgs)
            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        # Turn total_correct & total_samples into a tensor, so we can use all_reduce to sum them
        # across processes
        tensor = t.tensor([total_correct, total_samples], device=self.device)
        all_reduce(tensor, self.rank, self.args.world_size, op="sum")
        total_correct, total_samples = tensor.tolist()

        accuracy = total_correct / total_samples
        if self.rank == 0:
            wandb.log({"accuracy": accuracy}, step=self.examples_seen)
        return accuracy

    def train(self):
        self.pre_training_setup()

        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            t0 = time.time()

            if self.args.world_size > 1:
                self.train_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)

            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training", disable=self.rank != 0)
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen=:06}", refresh=False)

            accuracy = self.evaluate()

            if self.rank == 0:
                wandb.log({"epoch_duration": time.time() - t0}, step=self.examples_seen)

        if self.rank == 0:
            wandb.finish()
            t.save(self.model.state_dict(), f"resnet_{self.rank}.pth")


def dist_train_resnet_from_scratch(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size, use_wandb=False)  # flip to True to log to wandb
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


# ===== Bonus - DDP =====

def run(rank: int, world_size: int):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")
    model = DDP(SimpleModel().to(device), device_ids=[rank])  # Wrap the model with DDP
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)

    input = t.tensor([rank], dtype=t.float32, device=device)
    output = model(input)
    loss = output.pow(2).sum()
    loss.backward()  # DDP handles gradient synchronization

    optimizer.step()
    print(f"Rank {rank}, new param: {model.module.param.data}")

    dist.destroy_process_group()


# ===== Bonus - ring operations =====


def ring_all_reduce(tensor: Tensor, rank, world_size, op: Literal["sum", "mean"] = "sum") -> None:
    """
    Ring all_reduce implementation using non-blocking send/recv to avoid deadlock.
    """
    # Clone the tensor as the "send_chunk" for initial accumulation
    send_chunk = tensor.clone()

    # Step 1: Reduce-Scatter phase
    for _ in range(world_size - 1):
        # Compute the ranks involved in this round of sending/receiving
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1 + world_size) % world_size

        # Prepare a buffer for the received chunk
        recv_chunk = t.zeros_like(send_chunk)

        # Non-blocking send and receive
        send_req = dist.isend(send_chunk, dst=send_to)
        recv_req = dist.irecv(recv_chunk, src=recv_from)
        send_req.wait()
        recv_req.wait()

        # Accumulate the received chunk into the tensor
        tensor += recv_chunk

        # Update send_chunk for the next iteration
        send_chunk = recv_chunk

    # Step 2: All-Gather phase
    send_chunk = tensor.clone()
    for _ in range(world_size - 1):
        # Compute the ranks involved in this round of sending/receiving
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1 + world_size) % world_size

        # Prepare a buffer for the received chunk
        recv_chunk = t.zeros_like(send_chunk)

        # Non-blocking send and receive, and wait for completion
        send_req = dist.isend(send_chunk, dst=send_to)
        recv_req = dist.irecv(recv_chunk, src=recv_from)
        send_req.wait()
        recv_req.wait()

        # Update the tensor with received data
        tensor.copy_(recv_chunk)

        # Update send_chunk for the next iteration
        send_chunk = recv_chunk

    # Step 3: Average the final result
    if op == "mean":
        tensor /= world_size


# ===== Run the exercises (comment out any line to skip it) =====

if __name__ == "__main__":
    IN_COLAB = "google.colab" in sys.modules
    assert not IN_COLAB, "Should be doing these exercises in VS Code"
    assert t.cuda.is_available()
    assert t.cuda.device_count() > 1, "This example requires at least 2 GPUs per machine"

    # Basic send & receiving: CPU processes (gloo), then GPUs (nccl)
    # mp.spawn(send_receive, args=(2,), nprocs=2, join=True)
    # mp.spawn(send_receive_nccl, args=(2,), nprocs=2, join=True)

    # # Exercise - implement `broadcast`
    #tests.test_broadcast(broadcast, WORLD_SIZE)

    # # Exercise - implement `all_reduce`
    # tests.test_reduce(reduce, WORLD_SIZE)
    # tests.test_all_reduce(all_reduce, WORLD_SIZE)

    # # Syncing the gradients of a simple model with all_reduce
    # mp.spawn(run_simple_model, args=(2,), nprocs=2, join=True)

    # # Exercise - complete `DistResNetTrainer` (full training on all GPUs)
    # mp.spawn(dist_train_resnet_from_scratch, args=(t.cuda.device_count(),), nprocs=t.cuda.device_count(), join=True)

    # # Bonus - DDP
    # mp.spawn(run, args=(2,), nprocs=2, join=True)

    # # Bonus - ring operations
    tests.test_all_reduce(ring_all_reduce, WORLD_SIZE)
