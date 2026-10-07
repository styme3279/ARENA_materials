import torch as t
import torch.distributed as dist


def send_receive_nccl(rank, world_size):
    t.cuda.set_device(rank)
    device = t.device(f"cuda:{rank}")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    try:
        if rank == 0:
            tensor = t.zeros(1, device=device)
            dist.send(tensor, dst=1)
            print(f"GPU {rank} sent {tensor}", flush=True)

        elif rank == 1:
            tensor = t.ones(1, device=device)
            dist.recv(tensor, src=0)
            print(f"GPU {rank} received {tensor}", flush=True)

    finally:
        dist.destroy_process_group()