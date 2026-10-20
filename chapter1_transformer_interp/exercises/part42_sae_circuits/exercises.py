# %%
import gc
import os
import sys
from collections import Counter, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, TypeAlias

import einops
import numpy as np
import plotly.express as px
import torch as t
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from IPython.display import IFrame, display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from sae_lens import SAE, ActivationsStore, HookedSAETransformer
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from tabulate import tabulate
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt, to_numpy

dtype = t.float32  # t.bfloat16
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
device = str(device)  # SAELens expects device as string; this is easier!


def _get_hook_layer(sae: SAE) -> int:
    """Extract the layer number from an SAE's hook name (e.g. 'blocks.7.hook_resid_pre' → 7)."""
    return int(sae.cfg.metadata.hook_name.split(".")[1])


# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part42_sae_circuits"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part42_sae_circuits.tests as tests
import part42_sae_circuits.utils as utils

MAIN = __name__ == "__main__"

# %%
gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device, dtype=dtype)

gpt2_saes = {
    layer: SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=device,
        dtype=dtype,
    )
    for layer in tqdm(range(gpt2.cfg.n_layers))
}

# %%
