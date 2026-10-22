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
    %pip install transformers peft einops jaxtyping torchinfo git+https://github.com/ARENA-education/eindex.git git+https://github.com/ARENA-education/liveplot.git

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
import contextlib
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
