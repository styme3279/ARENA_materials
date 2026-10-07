import os
import torch.distributed as dist
from torch import Tensor
import part3_optimization.tests as tests

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

WORLD_SIZE = 2

def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    pass

if __name__ == "__main__":
    tests.test_broadcast(broadcast, WORLD_SIZE)