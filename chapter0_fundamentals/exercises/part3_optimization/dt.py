import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from IPython.core.display import HTML
from IPython.display import display
from torch import Tensor, optim

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
        tensors = []
        for r in range(world_size):
            if r != dst:
                received_tensor = t.zeros(tensor.shape)
                dist.recv(tensor=received_tensor, src=r)
                tensors.append(received_tensor)

        result: Tensor = None

        if op == "sum":
            result = t.sum(t.tensor(tensors, device=tensor.device))
        elif op =="mean":
            result = t.mean(t.tensor(tensors, device=tensor.device))

        tensor.copy_(result)

        

    else:
        dist.send(tensor=tensor, dst=dst)


def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    raise NotImplementedError()


if __name__ == "__main__":

    tests.test_reduce(reduce, WORLD_SIZE)
    tests.test_all_reduce(all_reduce, WORLD_SIZE)
