# %%
import contextlib
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from liveplot import LivePlot
from tqdm.auto import tqdm

# Must be set before torch initialises CUDA; reduces allocator
# fragmentation, which matters when the trainer is running near the 16GB limit of an A4000.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch as t
from eindex import eindex
from jaxtyping import Float, Int
from peft import PeftModel
from torch import Tensor
from transformers import PreTrainedTokenizerBase

# Make sure exercises are in the path
chapter = "chapter2_rl"
section = "part42_rlvr"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part42_rlvr import tests, utils

# With PyTorch's default of one thread per core, on machines that report many more cores than they let you use
# (common in cloud containers), CPU-side work spends longer waiting on threads than computing. Adjust if needed.
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Optional speed-up: Liger's fused RMSNorm / SwiGLU / RoPE kernels make the learning phase ~15-25% faster (same results up
# to bf16 rounding). They're Triton kernels, so only on GPUs where those work (pre-Blackwell); otherwise we skip them.
try:
    if device.type == "cuda" and t.cuda.get_device_capability()[0] < 10:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3(rope=True, rms_norm=True, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=False)
        print("Applied Liger kernels!")
except ImportError:
    print("Skipping Liger kernels...")
    pass


MAIN = __name__ == "__main__"
SLOW = False  # set True to run the slow optional experiments (the pass@k comparison takes ~20 min on an A40)

# %%
