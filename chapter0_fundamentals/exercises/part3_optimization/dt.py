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
    # dist.init_process_group(backend="ncll", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")

    for r in [0, 1]:
        if r != src:
            print(f"{rank=}, {device=}, sending {tensor=}")
            dist.send(tensor, dst=r)
    

    dist.destroy_process_group()


if __name__ == "__main__":

    tests.test_broadcast(broadcast, WORLD_SIZE)
