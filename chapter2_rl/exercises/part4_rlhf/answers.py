# %%
import gc
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal

# Allow expandable memory segments on CUDA to avoid OOMs from fragmentation (the IMDB run on gpt2-medium needs ~21GB
# of GPU memory without this, ~18GB with it)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
import wandb
from eindex import eindex
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from tabulate import tabulate
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

# Make sure exercises are in the path
chapter = "chapter2_rl"
section = "part4_rlhf"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part4_rlhf import tests, tests_lora  # , tl_ext
from part4_rlhf.fast_gen import fast_generate

# TransformerLens processes a model's weights on the CPU when it loads it. With PyTorch's default of one thread
# per core, on machines that report many more cores than they let you use (common in cloud containers), threads
# spend longer waiting than computing: loading gpt2-medium took ~85s instead of ~7s. Adjust if needed.
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
# TF32 matmuls on NVIDIA GPUs: ~1.5x faster RLHF training (no effect on CPU, MPS or pre-Ampere GPUs)
t.set_float32_matmul_precision("high")


MAIN = __name__ == "__main__"

# %%
# LOW_GPU_MEM fires automatically on GPUs with less than 22 GiB (e.g. Colab's 16GB T4), or with no CUDA GPU at all.
# gpt2-medium peaks at ~18 GiB on today's runs, so it fits 24GB cards (RTX 4090, A10, L4 report 22.5-24 GiB).

LOW_GPU_MEM = not t.cuda.is_available() or t.cuda.get_device_properties(device).total_memory / (1024 ** 3) < 22
BASE_MODEL = "gpt2-small" if LOW_GPU_MEM else "gpt2-medium"
RUN_BASE_RLHF = True
print(f"{LOW_GPU_MEM=}, {BASE_MODEL=}, {RUN_BASE_RLHF=}")

# %%
@dataclass
class RLHFArgs:
    # Basic / global
    seed: int = 1

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "RLHF"
    wandb_entity: str | None = None

    # Duration of different phases
    total_phases: int = 100
    batch_size: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 2

    # Optimization hyperparameters
    base_lr: float = 2e-5
    head_lr: float = 5e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 20
    final_scale: float = 0.1

    # Computing other PPO loss functions
    clip_coef: float = 0.2
    vf_coef: float = 0.15

    # Base model & sampling arguments
    base_model: str = BASE_MODEL
    gen_len: int = 30
    temperature: float = 1.0
    top_k: int = 10
    prefix: str = "This is"
    prepend_bos: bool = True

    # RLHF-specific arguments
    kl_coef: float = 2.5
    reward_fn: Callable = lambda samples: t.zeros(len(samples), device=device)
    normalize_reward: bool = True

    def __post_init__(self):
        assert self.total_phases > self.warmup_steps, "total_phases must be greater than warmup_steps"
        assert self.batch_size % self.num_minibatches == 0, "batch_size should be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches

# %%
class HookedTransformerWithValueHead(HookedTransformer):
    """
    Defines a GPT model with a value head (the latter taking the last hidden state as input, post-layernorm).

    The value head is a simple MLP with one hidden layer, and scalar output:

        Linear(d_model -> 4*d_model)
        ReLU
        Linear(4*d_model -> 1)

    All linear layers have biases.
    """

    value_head: nn.Sequential
    value_head_output: Float[Tensor, "batch seq"]
    value_head_hook: tuple[str, Callable]

    @classmethod
    def from_pretrained(cls, *args, use_value_head: bool = True, **kwargs):
        # Build the model directly on `device`: otherwise TransformerLens builds it and processes its weights on the
        # CPU, then moves it (about 2x slower to load)
        with t.device(kwargs.setdefault("device", device)):
            model = super(HookedTransformerWithValueHead, cls).from_pretrained(*args, **kwargs)
        model.value_head_hook = ("ln_final.hook_normalized", model.run_value_head)

        raise NotImplementedError()

    @property
    def fwd_hooks(self):
        return [self.value_head_hook]

    def get_base_model_trainable_params(self):
        return (p for name, p in self.named_parameters() if "value_head" not in name)

    def get_value_head_params(self):
        return self.value_head.parameters()

    def run_value_head(self, resid_post: Float[Tensor, "batch seq d_model"], hook: HookPoint):
        raise NotImplementedError()

    def forward_with_value_head(
        self,
        input_ids: Int[Tensor, "batch seq"],
        **kwargs,
    ) -> tuple[Float[Tensor, "batch seq d_vocab"], Float[Tensor, "batch seq"]]:
        raise NotImplementedError()


model = HookedTransformerWithValueHead.from_pretrained("pythia-14m", use_value_head=True).to(device)
tests.test_transformer_with_value_head(model)
