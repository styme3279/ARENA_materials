# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter2_rl"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
try:
    import jaxtyping
except:
    %pip install transformers einops jaxtyping wandb tabulate git+https://github.com/ARENA-education/eindex.git

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
    if not IN_COLAB:
        !sudo apt-get install unzip
        %pip install jupyter ipython --upgrade

    if not os.path.exists(f"{root}/{chapter}"):
        !wget -P {root} https://github.com/ARENA-education/ARENA_materials/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/{repo}-{branch}


if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")
# %%
import gc
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Literal

# Allow expandable memory segments on CUDA to avoid OOMs from fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
import wandb
from jaxtyping import Float, Int
from tabulate import tabulate
from torch import Tensor
from tqdm.auto import tqdm
from eindex import eindex
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Make sure exercises are in the path
chapter = "chapter2_rl"
section = "part4_rlhf"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

from part4_rlhf import tests, tests_lora

# PyTorch defaults to one thread per core. On machines that report many more cores than they let you use (common in
# cloud containers), CPU-side work then spends longer waiting on threads than computing. Adjust if needed.
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
# TF32 matmuls on NVIDIA GPUs: noticeably faster training (no effect on CPU, MPS or pre-Ampere GPUs)
t.set_float32_matmul_precision("high")


MAIN = __name__ == "__main__"
# %%
# LOW_GPU_MEM fires automatically on GPUs with less than 22 GiB (e.g. Colab's 16GB T4), or with no CUDA GPU at all.
LOW_GPU_MEM = not t.cuda.is_available() or t.cuda.get_device_properties(device).total_memory / (1024**3) < 22
BASE_MODEL = "gpt2" if LOW_GPU_MEM else "gpt2-medium"
LORA_MODEL = "Qwen/Qwen3-0.6B-Base" if LOW_GPU_MEM else "Qwen/Qwen3-1.7B-Base"
# bfloat16 halves the memory of the frozen base weights in the LoRA section; GPUs before Ampere (e.g. T4) don't support it
LORA_DTYPE = t.bfloat16 if t.cuda.is_available() and t.cuda.is_bf16_supported() else t.float32
RUN_BASE_RLHF = True
print(f"{LOW_GPU_MEM=}, {BASE_MODEL=}, {LORA_MODEL=}, {LORA_DTYPE=}, {RUN_BASE_RLHF=}")

# %%
@dataclass
class GRPO_Args:
    # Basic / global
    seed: int = 1

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "RLHF"
    wandb_entity: str | None = None

    # Duration of different phases
    total_phases: int = 100
    prompt: str = "This is"
    batch_size: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 2

    # Optimization hyperparameters
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    final_scale: float = 0.1

    # GRPO objective
    clip_coef: float = 0.2
    kl_coef: float = 0.3

    # Base model & sampling arguments
    base_model: str = BASE_MODEL
    gen_len: int = 30
    temperature: float = 1.0

    # Reward function, mapping a list of samples (prompt + completion) to a tensor of rewards
    reward_fn: Callable = lambda samples: t.zeros(len(samples), device=device)

    def __post_init__(self):
        assert self.total_phases > self.warmup_steps, "total_phases must be greater than warmup_steps"
        assert self.batch_size % self.num_minibatches == 0, "batch_size should be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches

# %%
@t.no_grad()
def get_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    batch_size: int,
    gen_len: int,
    temperature: float = 1.0,
) -> tuple[Int[Tensor, "batch seq"], list[str]]:
    """
    Samples `batch_size` completions of `gen_len` tokens, all from the same prompt.

    Args:
        model: the language model to sample from
        tokenizer: the model's tokenizer
        prompt: the prompt to complete
        batch_size: the number of completions to sample
        gen_len: the number of new tokens to generate for each completion
        temperature: the sampling temperature (1.0 means sampling from the model's distribution)

    Returns:
        sample_ids: the token ids of the samples (prompt + completion). The completions are the last `gen_len` tokens.
        samples: the samples as strings (prompt + completion, without special tokens), used for the reward
    """
    prompt_ids = tokenizer(tokenizer.eos_token + prompt, return_tensors="pt").input_ids.to(device).repeat(batch_size, 1)
    sample_ids = model.generate(
        prompt_ids,
        # Without an explicit attention mask, `generate` treats every pad token in the prompt as padding. Our pad token
        # is the end-of-text token that starts the prompt, so it would mask that out and sample from a different
        # context to the one we compute log-probabilities on.
        attention_mask=t.ones_like(prompt_ids),
        do_sample=True,
        temperature=temperature,
        top_k=0,  # no truncation: sample from the model's full distribution
        top_p=1.0,
        max_new_tokens=gen_len,
        eos_token_id=None,  # don't stop at the end-of-text token: always generate exactly gen_len tokens
        pad_token_id=tokenizer.eos_token_id,
    )

    completions = tokenizer.batch_decode(sample_ids[:, -gen_len:], skip_special_tokens=True)
    samples = []
    for completion in completions:
        samples.append(prompt + completion)
    return sample_ids, samples

# %%
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=t.float32).to(device)

sample_ids, samples = get_samples(model, tokenizer, prompt="So long, and thanks for all the", batch_size=5, gen_len=15)

table = [[str(ids.tolist()), repr(sample)] for ids, sample in zip(sample_ids, samples)]
print(tabulate(table, ["Token ids", "Sample"], tablefmt="simple_grid", maxcolwidths=60))

# %%
def reward_fn_char_count(generated_sample: list[str], char: str = ".") -> Float[Tensor, " batch"]:
    """
    Reward function counting the number of instances of a particular character.

    Args:
        generated_sample: the generated samples to evaluate
        char: the character to count

    Returns:
        A float tensor of shape (batch,) giving the count of `char` in each sample.
    """
    counts = t.zeros(len(generated_sample), dtype=t.float32)
    for s in generated_sample:
        char_count = s.count(char)


# Test your reward function
A = "This is a test."
B = "......"
C = "Whatever"

t.testing.assert_close(reward_fn_char_count([A]), t.tensor([1.0], device=device))
t.testing.assert_close(reward_fn_char_count([A, B, C]), t.tensor([1.0, 6.0, 0.0], device=device))
t.testing.assert_close(reward_fn_char_count([A], " "), t.tensor([3.0], device=device))
print("All tests for `reward_fn_char_count` passed!")