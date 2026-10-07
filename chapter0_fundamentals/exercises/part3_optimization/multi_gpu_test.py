# %%
import os
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp

MAIN = __name__ == "__main__"

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


if MAIN:
    world_size = 2  # simulate 2 processes
    mp.spawn(
        send_receive,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

assert t.cuda.is_available()
assert t.cuda.device_count() > 1, "This example requires at least 2 GPUs per machine"
# %%

# %%
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


if MAIN:
    world_size = 2  # simulate 2 processes
    mp.spawn(
        send_receive_nccl,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
# %%

# %%

def run_broadcast(rank: int, world_size: int, broadcast):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    t.cuda.set_device(rank)

    # Create a tensor for each rank with its rank as the value
    tensor = t.tensor([float(rank)], dtype=t.float32).cuda()

    # Run broadcast operation (tensor is broadcasted from rank 0 to all ranks)
    broadcast(tensor, rank, world_size, src=0)

    # Check and print results on all ranks
    print(f"Rank {rank} broadcasted tensor: expected 0.0 (from rank 0), got {tensor}")
    t.testing.assert_close(tensor, t.full_like(tensor, 0.0))

def test_broadcast(broadcast, world_size):
    world_size = world_size  # Number of processes (simulated ranks)
    mp.spawn(run_broadcast, args=(world_size, broadcast), nprocs=world_size, join=True)
    print("All tests in `test_broadcast` passed!")







if MAIN:
    test_broadcast(broadcast, WORLD_SIZE)
# %%
