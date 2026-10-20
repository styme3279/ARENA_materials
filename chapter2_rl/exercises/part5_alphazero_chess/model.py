"""
The AlphaZero network for chess: the same shape as `Connect4Model` (shared residual trunk -> value head
+ policy head), scaled up to 19 input planes and a (73, 8, 8) = 4672-way policy.

The policy head stays convolutional (as in the AlphaZero paper): plane k at square s scores "move type
k starting from s", so the flattened output index is k * 64 + s -- exactly `chess_env`'s action index.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_env import N_IN_PLANES, N_MOVE_PLANES


class ResBlock(nn.Module):
    """x -> conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> (+ x) -> ReLU (identical to the Connect-4 block)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(h)) + x)


class ChessNet(nn.Module):
    def __init__(self, channels: int = 96, blocks: int = 6, value_hidden: int = 128):
        super().__init__()
        self.config = dict(channels=channels, blocks=blocks, value_hidden=value_hidden)
        self.stem = nn.Sequential(
            nn.Conv2d(N_IN_PLANES, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.Flatten(),
            nn.Linear(8 * 64, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1), nn.Tanh())
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, N_MOVE_PLANES, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 19, 8, 8) float canonical planes -> value (B,) in [-1, 1], logits (B, 4672)."""
        h = self.trunk(self.stem(x))
        return self.value_head(h).squeeze(-1), self.policy_head(h).flatten(1)


def to_input(obs_u8: torch.Tensor) -> torch.Tensor:
    """uint8 planes (B, 19, 8, 8) -> float network input (the half-move clock plane is scaled by 1/100)."""
    x = obs_u8.float()
    x[:, 18] *= 0.01
    return x


def legal_logits(logits: torch.Tensor, lidx: torch.Tensor) -> torch.Tensor:
    """Gather the logits of each position's legal moves. lidx: (B, L) long, -1 padded -> (B, L), pads = -inf."""
    g = logits.gather(1, lidx.clamp_min(0))
    return g.masked_fill(lidx < 0, float("-inf"))


@torch.no_grad()
def infer(model: nn.Module, obs_u8: torch.Tensor, lidx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One inference pass: (values (B,), priors (B, L)) where priors = softmax over each row's legal moves."""
    with torch.autocast(device_type=obs_u8.device.type, dtype=torch.bfloat16, enabled=obs_u8.is_cuda):
        value, logits = model(to_input(obs_u8))
    return value.float(), torch.softmax(legal_logits(logits.float(), lidx), dim=-1)


class GraphedInference:
    """`infer` replayed from CUDA graphs captured at a few padded batch sizes.

    The eager forward of a 10-block net dispatches ~150 kernels through autocast, costing a few ms of
    *CPU* per call; at hundreds of calls per second that alone eats a core of the CPU quota the self-play
    workers need. A graph replay costs microseconds. Weights are read in place at replay time, so
    training updates are picked up automatically -- which is why autocast's weight-cast cache must be
    off during capture (otherwise the graph would keep using bf16 copies of the *old* weights).
    """

    def __init__(self, model: nn.Module, device, buckets=(64, 128, 256, 512, 1024, 2048), max_legal: int = 256):
        self.model, self.device, self.buckets, self.max_legal = model, device, buckets, max_legal
        self.graphs = {}
        pool = None
        was_training = model.training
        model.eval()
        for B in sorted(buckets, reverse=True):                 # largest first, then share its memory pool
            obs = torch.zeros((B, N_IN_PLANES, 8, 8), dtype=torch.uint8, device=device).contiguous(
                memory_format=torch.channels_last)
            obs[:, 0, 0, 0] = 1
            lidx = torch.zeros((B, max_legal), dtype=torch.long, device=device)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(3):
                    self._run(obs, lidx)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g, pool=pool):
                v, p = self._run(obs, lidx)
            pool = g.pool()
            self.graphs[B] = (g, obs, lidx, v, p)
        model.train(was_training)
        self.done = torch.cuda.Event(blocking=True)

    def _run(self, obs, lidx):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
            value, logits = self.model(to_input(obs))
        return value.float(), torch.softmax(legal_logits(logits.float(), lidx), dim=-1)

    def run(self, obs: torch.Tensor, lidx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tensor version for callers already on the GPU: obs (n, 19, 8, 8) uint8, lidx (n, L) long (-1
        padded, L <= max_legal) -> values (n,), priors (n, L) (fresh tensors, safe to keep)."""
        n, L = obs.shape[0], lidx.shape[1]
        if n > self.buckets[-1]:
            parts = [self.run(obs[s:s + self.buckets[-1]], lidx[s:s + self.buckets[-1]])
                     for s in range(0, n, self.buckets[-1])]
            return torch.cat([a for a, _ in parts]), torch.cat([b for _, b in parts])
        B = next(b for b in self.buckets if b >= n)
        g, sobs, slidx, v, p = self.graphs[B]
        sobs[:n].copy_(obs)
        slidx[:n, :L].copy_(lidx)
        if L < self.max_legal:
            slidx[:n, L:] = -1
        g.replay()
        return v[:n].clone(), p[:n, :L].clone()

    def __call__(self, obs_np, lidx_np):
        """numpy uint8 (n, 19, 8, 8), int (n, L) -1 padded -> numpy values (n,), priors (n, L)."""
        n, L = obs_np.shape[0], lidx_np.shape[1]
        if n > self.buckets[-1]:
            parts = [self(obs_np[s:s + self.buckets[-1]], lidx_np[s:s + self.buckets[-1]])
                     for s in range(0, n, self.buckets[-1])]
            return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])
        B = next(b for b in self.buckets if b >= n)
        g, obs, lidx, v, p = self.graphs[B]
        obs[:n].copy_(torch.from_numpy(obs_np))
        lidx[:n, :L].copy_(torch.from_numpy(lidx_np))
        if L < self.max_legal:
            lidx[:n, L:] = -1
        g.replay()
        self.done.record()
        self.done.synchronize()                                 # sleep, don't spin, until the GPU is done
        return v[:n].cpu().numpy(), p[:n, :L].cpu().numpy()


def az_loss(value, logits, lidx, pi, z, value_coef: float = 1.0):
    """AlphaZero loss with the policy softmax restricted to legal moves (as at inference time).

    value (B,), logits (B, 4672), lidx (B, L) long (-1 padded), pi (B, L) visit distribution aligned
    with lidx, z (B,) game outcome for the side to move. Returns (total, policy_ce, value_mse).
    """
    logp = F.log_softmax(legal_logits(logits.float(), lidx), dim=-1).masked_fill(lidx < 0, 0.0)
    policy = -(pi * logp).sum(-1).mean()
    value_mse = F.mse_loss(value.float(), z)
    return policy + value_coef * value_mse, policy, value_mse


def save_checkpoint(path, model, extra: dict | None = None):
    torch.save({"model": model.state_dict(), "model_config": model.config, **(extra or {})}, path)


def load_model(path, device="cpu") -> tuple[ChessNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ChessNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
