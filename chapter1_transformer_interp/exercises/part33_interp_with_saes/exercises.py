# %%

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
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part33_interp_with_saes.tests as tests
import part33_interp_with_saes.utils as utils

MAIN = __name__ == "__main__"

# %%

sae_release = "gpt2-small-res-jb-feature-splitting"

widths = [768 * (2**n) for n in range(7)]  # Note, you can increase to 8 if it fits on your GPU
sae_ids = [f"blocks.8.hook_resid_pre_{width}" for width in widths]

splitting_saes = {
    width: SAE.from_pretrained(sae_release, sae_id, device=str(device)) for width, sae_id in zip(widths, sae_ids)
}

gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

# %%
def load_and_process_autointerp_dfs(width: int):
    # Load in dataframe, from the gzipped JSONL files which Neuronpedia publishes in a public S3 bucket (see
    # https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/ - there's one file per ~1024
    # latents, so we list all the files for this SAE then download & concatenate them)
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[f"blocks.8.hook_resid_pre_{width}"]  # e.g. "gpt2-small/8-res_fs768-jb"
    params = {"list-type": "2", "prefix": f"v1/{neuronpedia_id}/explanations/"}
    listing = requests.get("https://neuronpedia-datasets.s3.us-east-1.amazonaws.com", params=params)
    listing.raise_for_status()
    keys = re.findall(r"<Key>(.*?)</Key>", listing.text)
    assert len(keys) > 0, f"No explanations found on Neuronpedia for {neuronpedia_id!r}"
    df = pd.concat(
        [
            pd.read_json(f"https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/{key}", lines=True, compression="gzip")[["index", "description"]]
            for key in tqdm(keys, desc=f"Downloading explanations for {width=}")
        ],
        ignore_index=True,
    )

    # Drop duplicate latent descriptions
    df["index"] = df["index"].astype(int)
    df = df.drop_duplicates(subset=["index"], keep="first").sort_values("index", ignore_index=True)

    # Fill in missing latent descriptions with empty strings
    full_index = pd.DataFrame({"index": range(width)})
    df = full_index.merge(df, on="index", how="left")
    df["description"] = df["description"].fillna("")
    print(f"Loaded autointerp df for {width=}")
    if (n_missing := (df["description"] == "").sum()) > 0:
        print(f"  Warning: {n_missing}/{len(df)} latents missing descriptions")

    return df


autointerp_dfs = {width: load_and_process_autointerp_dfs(width) for width in widths}
display(autointerp_dfs[768].head())

# %%
from rich import inspect

def display_dashboard(
    sae_release=sae_release,
    sae_id="blocks.7.hook_resid_pre",
    latent_idx=0,
    width=800,
    height=600,
):
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]

    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

    print(url)
    display(IFrame(url, width=width, height=height))

rprint(sae_ids)
latent_idx = 34
sae_id = sae_ids[0]
display_dashboard(
    sae_id=sae_id,
    latent_idx=latent_idx,
    )

# %%
