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

# rprint(sae_ids)
# latent_idx = 34
# display_dashboard(
#     sae_id=sae_id,
#     latent_idx=latent_idx,
#     )

# %%

# %%

#### FIND MAX ACTIVATING TOKENS FOR GIVEN PROMPT #####

prompt = "The cat is near the table in the living room"
# answer = " priority"

def get_top_k_from_sae(sae_to_use=0, k=3, model=gpt2):
    sae_id = sae_ids[sae_to_use]
    sae = splitting_saes[int(list(splitting_saes.keys())[sae_to_use])]
    print(f"Running sae: {sae_id}")

    # Get top activations on final token
    _, cache = model.run_with_cache_with_saes(
        prompt,
        saes=[sae],
        stop_at_layer=_get_hook_layer(sae) + 1,
    )

    # cache is [batch, seq_len, d_sae]
    sae_acts_post = cache[f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"][0, -1, :]

    # Plot line chart of latent activations
    px.line(
        sae_acts_post.cpu().numpy(),
        title=f"Latent activations at the final token position ({sae_acts_post.nonzero().numel()} alive)",
        labels={"index": "Latent", "value": "Activation"},
        width=1000,
    ).update_layout(showlegend=False).show()

    top_latents = []
    # Print the top 5 latents, and inspect their dashboards
    for act, ind in zip(*sae_acts_post.topk(k)):
        print(f"Latent {ind} had activation {act:.2f}")
        top_latents.append((ind, act))

    return top_latents

top_latents_for_sae = []
for i in range(len(sae_ids)):
    top_latents = get_top_k_from_sae(sae_to_use=i, k=3)
    top_latents_for_sae.append((i, top_latents))

rprint(top_latents_for_sae)
    
# %%

latent_ids = [295, 468, 645]   # 0
latent_ids = [951, 1867, 10139] # -2
for latent_idx in latent_ids:
    sae_id     = -2
    display_dashboard(
        sae_id     = sae_ids[sae_id],
        latent_idx = latent_idx, 
    )

# %%










# # First see how the model does without SAEs
# test_prompt(prompt, answer, gpt2)

# rprint(splitting_saes)
# # Test our prompt, to see what the model says
# with gpt2.saes(saes=[sae]):
#     test_prompt(prompt, answer, gpt2)

# # Same thing, done in a different way
# gpt2.add_sae(sae)
# test_prompt(prompt, answer, gpt2)
# gpt2.reset_saes()  # Remember to always do this!

# # Using `run_with_saes` method in place of standard forward pass
# logits = gpt2(prompt, return_type="logits")
# logits_sae = gpt2.run_with_saes(prompt, saes=[sae], return_type="logits")
# answer_token_id = gpt2.to_single_token(answer)

# # Getting model's prediction
# top_prob, token_id_prediction = logits[0, -1].softmax(-1).max(-1)
# top_prob_sae, token_id_prediction_sae = logits_sae[0, -1].softmax(-1).max(-1)

# print(f"""Standard model:
#     top prediction = {gpt2.to_string(token_id_prediction)!r}
#     prob = {top_prob.item():.2%}
# SAE reconstruction:
#     top prediction = {gpt2.to_string(token_id_prediction_sae)!r}
#     prob = {top_prob_sae.item():.2%}
# """)

# # %%
# _, cache = gpt2.run_with_cache_with_saes(prompt, saes=[sae])

# for name, param in cache.items():
#     if "hook_sae" in name:
#         print(f"{name:<43}: {tuple(param.shape)}")


# %%
import hdbscan
import numba
from umap import UMAP

numba.config.THREADING_LAYER = "workqueue"


def compute_sae_umap_data(
    saes: dict[int, SAE],
    autointerp_dfs: dict[int, pd.DataFrame],
    sae_widths: list[int],
    model: HookedSAETransformer,
    n_neighbors_visual: int = 15,
    min_dist_visual: float = 0.05,
    find_clusters: bool = False,
    n_neighbors_cluster: float = 15,
    min_dist_cluster: float = 0.1,
    min_cluster_size: int = 3,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """
    This function will return a dataframe containing umap coordinates & other data (you can then use
    this to create a plot using the code immediately below). The UMAP calculation is done over
    multiple SAEs simultaneously, for comparison.

    Expected dataframe columns:
        sae_width: int
            The width of the SAE that this feature belongs to
        feature_idx: int
            The index of the feature
        umap_x: float
            The x-coordinate of the feature in the UMAP embedding
        umap_y: float
            The y-coordinate of the feature in the UMAP embedding
        autointerp: str
            The autointerp description of this feature
        top_token_strs_formatted: str
            The top 10 tokens that the feature is activated by

    Args:
        saes: dict[int, SAE]
            List of SAEs to use for the UMAP calculation
        autointerp_dfs: dict[int, pd.DataFrame]
            Dataframes containing autointerp descriptions for each SAE
        sae_widths: list[int]
            The widths of SAEs we'll be using for the UMAP calculation
        model: HookedSAETransformer
            The model which all the SAEs should be attached to
        n_neighbors_visual: int
            The number of neighbors to consider for the UMAP embedding for the visual plot
        min_dist_visual: float
            The minimum distance between points in the UMAP embedding for the visual plot
        n_neighbors_cluster: int
            The number of neighbors to consider for the UMAP embedding for the cluster plot
        min_dist_cluster: float
            The minimum distance between points in the UMAP embedding for the cluster plot
        min_cluster_size: int
            The minimum number of points in a cluster.
        batch_size: int
            Number of features to process at once, for logits
    """
    # assert not find_clusters, "Not implemented yet"

    # Get initial dataframe by concatenating across SAEs (and autointerp descriptions)
    sae_dfs = []
    for width in sae_widths:
        df = autointerp_dfs[width].copy()
        df["sae_width"] = width
        df["feature_idx"] = list(range(width))
        sae_dfs.append(df)
    feature_df = pd.concat(sae_dfs)

    # Get concatenated decoder matrix
    W_dec = t.cat([saes[width].W_dec for width in sae_widths])

    # Get all the top boosted tokens for each feature, processing in batches
    top_token_ids = []
    print("Computing top logits")
    for start_idx in range(0, len(feature_df), batch_size):
        end_idx = min(start_idx + batch_size, len(feature_df))
        batch_result = W_dec[start_idx:end_idx] @ model.W_U
        top_token_ids.append(batch_result.topk(10).indices)

    # Combine results from all batches, and get them into the dataframe
    token_factors_inds = t.cat(top_token_ids)
    feature_df["tok_token_ids"] = token_factors_inds.tolist()
    feature_df["top_token_strs"] = [
        ", ".join(map(repr, model.to_str_tokens(tokens)))
        for tokens in token_factors_inds
    ]

    print("Calculating 2D UMAP")
    visual_umap = UMAP(
        n_components=2,
        n_neighbors=n_neighbors_visual,
        min_dist=min_dist_visual,
        metric="cosine",
    )
    visual_umap_embedding = visual_umap.fit_transform(W_dec.detach().cpu())

    feature_df[["umap_x", "umap_y"]] = visual_umap_embedding[:, :2]

    if find_clusters:
        print("Calculating 10D UMAP")
        clustering_umap = UMAP(
            n_components=10,
            n_neighbors=n_neighbors_cluster,
            min_dist=min_dist_cluster,
            metric="cosine",
        )
        clustering_umap_embedding = clustering_umap.fit_transform(W_dec.detach().cpu())
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        clusterer.fit(clustering_umap_embedding)

        feature_df["cluster"] = clusterer.labels_
        feature_df.sort_values("cluster", inplace=True)
        feature_df["cluster"] = feature_df["cluster"].astype(str)

    return feature_df


# This took about 40s to run for me in Colab Pro+, 80s on my VastAI A100 remote machine
expansion_factors = [1, 4, 16]
umap_df = compute_sae_umap_data(splitting_saes, autointerp_dfs, [768 * ex for ex in expansion_factors], gpt2)
display(umap_df.head())

# %%

# %%
# %%
# Narrow → wide: first trace is drawn at the bottom, so the small (wide-SAE) points end up on top
widths_plotted = [768 * ex for ex in expansion_factors]
n = len(widths_plotted)

# Big points faint, small points opaque; distinct hues so each SAE is easy to tell apart
base_rgb = ["120,170,220", "245,150,60", "20,110,40", "150,30,120"]  # light blue, orange, dark green, purple
alphas = [0.25 + 0.75 * i / max(n - 1, 1) for i in range(n)]       # 0.25 → 1.0
color_map = {str(w): f"rgba({rgb},{a:.2f})" for w, rgb, a in zip(widths_plotted, base_rgb, alphas)}

# Marker size: larger for narrower SAEs (768 → 4, 3072 → 1, 12288 → 0.25)
umap_df["marker_size"] = 4 * umap_df["sae_width"].min() / umap_df["sae_width"]

px.scatter(
    umap_df,
    x="umap_x",
    y="umap_y",
    color=umap_df["sae_width"].astype(str),
    size="marker_size",
    height=900,
    width=1200,
    hover_data=["description", "top_token_strs"],
    labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2", "color": "SAE Width"},
    color_discrete_map=color_map,
    category_orders={"color": [str(w) for w in widths_plotted]},  # fixes trace (draw) order
    template="simple_white",
    title="Feature Splitting in SAEs",
).update_traces(marker=dict(line=dict(width=0))).show()
# %%


# This took about 50s to run for me in Colab Pro+, 90s on my VastAI A100 remote machine
umap_df = compute_sae_umap_data(splitting_saes, autointerp_dfs, [widths[5]], gpt2, find_clusters=True)
display(umap_df.head())

px.scatter(
    umap_df,
    x="umap_x",
    y="umap_y",
    color="cluster",
    height=900,
    width=1200,
    hover_data=["description", "top_token_strs"],
    labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2"},
    template="simple_white",
    title=f"2D UMAP for SAE width = {widths[5]}, clustering algorithm = HDBSCAN from 10D UMAP embedding",
).update_traces(marker=dict(size=4, line=dict(width=0))).update_layout(showlegend=False)

# %%














