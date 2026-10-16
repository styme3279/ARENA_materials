# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter1_transformer_interp"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
try:
    import transformer_lens
except:
    %pip install "openai>=1.56.2,<4" einops datasets jaxtyping "sae-lens>=6.51.3,<7.0.0" openai tabulate umap-learn hdbscan git+https://github.com/ARENA-education/eindex.git git+https://github.com/ARENA-education/CircuitsVis.git@arena-tl4.1#subdirectory=python git+https://github.com/ARENA-education/sae_vis.git@v0.3.7+arena.2 "transformer_lens>=3.9,<4"

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

import gc
import itertools
import os
import random
import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeAlias

import circuitsvis as cv
import einops
import pandas as pd
import plotly.express as px
import requests
import torch as t
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from IPython.display import HTML, IFrame, display
from jaxtyping import Float, Int
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from sae_lens import (
    SAE,
    ActivationsStore,
    GatedTrainingSAEConfig,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    LoggingConfig,
)
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from sae_vis import SaeVisConfig, SaeVisData, SaeVisLayoutConfig
from tabulate import tabulate
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


def _get_hook_layer(sae: SAE) -> int:
    """Extract the layer number from an SAE's hook name (e.g. 'blocks.7.hook_resid_pre' → 7)."""
    return int(sae.cfg.metadata.hook_name.split(".")[1])


# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part33_interp_with_saes"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

import part33_interp_with_saes.tests as tests
import part33_interp_with_saes.utils as utils

MAIN = __name__ == "__main__"
# %%

metadata_rows = [
    [data.model, data.release, data.repo_id, len(data.saes_map)] for data in get_pretrained_saes_directory().values()
]

# Print all SAE releases, sorted by base model
print(
    tabulate(
        sorted(metadata_rows, key=lambda x: x[0]),
        headers=["model", "release", "repo_id", "n_saes"],
        tablefmt="simple_outline",
    )
)

# %%
def format_value(value):
    return "{{{0!r}: {1!r}, ...}}".format(*next(iter(value.items()))) if isinstance(value, dict) and value else repr(value)


release = get_pretrained_saes_directory()["gpt2-small-res-jb"]

print(
    tabulate(
        [[k, format_value(v)] for k, v in release.__dict__.items()],
        headers=["Field", "Value"],
        tablefmt="simple_outline",
    )
)

# %%
data = [[id, path, release.neuronpedia_id[id]] for id, path in release.saes_map.items()]

print(
    tabulate(
        data,
        headers=["SAE id", "SAE path (HuggingFace)", "Neuronpedia ID"],
        tablefmt="simple_outline",
    )
)

# %%
t.set_grad_enabled(False)

gpt2: HookedSAETransformer = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

gpt2_sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.7.hook_resid_pre",
    device=str(device),
)

# %%
print(
    tabulate(
        list(gpt2_sae.cfg.__dict__.items()) + list(gpt2_sae.cfg.metadata.items()),
        headers=["name", "value"],
        tablefmt="simple_outline",
    )
)

# %%
def display_dashboard(
    sae_release="gpt2-small-res-jb",
    sae_id="blocks.7.hook_resid_pre",
    latent_idx=4527,
    width=800,
    height=600,
):
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]

    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

    print(url)
    display(IFrame(url, width=width, height=height))


latent_idx = random.randint(0, gpt2_sae.cfg.d_sae)
display_dashboard(latent_idx=latent_idx)

# %%
prompt = "Mitigating the risk of extinction from AI should be a global"
answer = " priority"

# First see how the model does without SAEs
test_prompt(prompt, answer, gpt2)

# Test our prompt, to see what the model says
with gpt2.saes(saes=[gpt2_sae]):
    test_prompt(prompt, answer, gpt2)

# Same thing, done in a different way
gpt2.add_sae(gpt2_sae)
test_prompt(prompt, answer, gpt2)
gpt2.reset_saes()  # Remember to always do this!

# Using `run_with_saes` method in place of standard forward pass
logits = gpt2(prompt, return_type="logits")
logits_sae = gpt2.run_with_saes(prompt, saes=[gpt2_sae], return_type="logits")
answer_token_id = gpt2.to_single_token(answer)

# Getting model's prediction
top_prob, token_id_prediction = logits[0, -1].softmax(-1).max(-1)
top_prob_sae, token_id_prediction_sae = logits_sae[0, -1].softmax(-1).max(-1)

print(f"""Standard model:
    top prediction = {gpt2.to_string(token_id_prediction)!r}
    prob = {top_prob.item():.2%}
SAE reconstruction:
    top prediction = {gpt2.to_string(token_id_prediction_sae)!r}
    prob = {top_prob_sae.item():.2%}
""")

# %%
