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
# gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device, dtype=dtype)

# gpt2_saes = {
#     layer: SAE.from_pretrained(
#         release="gpt2-small-res-jb",
#         sae_id=f"blocks.{layer}.hook_resid_pre",
#         device=device,
#         dtype=dtype,
#     )
#     for layer in tqdm(range(gpt2.cfg.n_layers))
# }

# %%
gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device, dtype=dtype)

hf_repo_id = "callummcdougall/arena-demos-transcoder"
sae_id = "gpt2-small-layer-{layer}-mlp-transcoder-folded-b_dec_out"
gpt2_transcoders = {
    layer: SAE.from_pretrained(release=hf_repo_id, sae_id=sae_id.format(layer=layer), device=device, dtype=dtype)
    for layer in tqdm(range(9))
}

layer = 8
gpt2_transcoder = gpt2_transcoders[layer]
print("Transcoder hooks (same as regular SAE hooks):", gpt2_transcoder.hook_dict.keys())

# Load the sparsity values, and plot them
log_sparsity_path = hf_hub_download(hf_repo_id, f"{sae_id.format(layer=layer)}/log_sparsity.pt")
log_sparsity = t.load(log_sparsity_path, map_location="cpu", weights_only=True)
fig = px.histogram(
    to_numpy(log_sparsity), width=800, template="ggplot2", title="Transcoder latent sparsity"
).update_layout(showlegend=False)
fig.show()


live_latents = np.arange(len(log_sparsity))[to_numpy(log_sparsity > -4)]

# Get the activations store
gpt2_act_store = ActivationsStore.from_sae(
    model=gpt2,
    sae=gpt2_transcoders[layer],
    dataset="NeelNanda/pile-10k",
    streaming=True,
    store_batch_size_prompts=16,
    n_batches_in_buffer=32,
    device=device,
)
tokens = gpt2_act_store.get_batch_tokens()
assert tokens.shape == (gpt2_act_store.store_batch_size_prompts, gpt2_act_store.context_size)





# %%
def run_with_cache_with_transcoder(
    model: HookedSAETransformer,
    transcoders: list[SAE],
    tokens: Tensor,
    use_error_term: bool = True,  # by default we don't intervene, just compute activations
) -> ActivationCache:
    """
    Runs MLP transcoder(s) on a batch of tokens, using native SAELens v6 transcoder support.

    If use_error_term=True (default), the model's activations are left intact and we just cache
    transcoder activations. If False, the model's MLP outputs are replaced with transcoder outputs.
    """
    prev_use_error_terms = [tc.use_error_term for tc in transcoders]
    for tc in transcoders:
        tc.use_error_term = use_error_term
    try:
        _, cache = model.run_with_cache_with_saes(tokens, saes=transcoders)
    finally:
        for tc, prev in zip(transcoders, prev_use_error_terms):
            tc.use_error_term = prev
    return cache

# %%
def get_k_largest_indices(
    x: Float[Tensor, "batch seq"], k: int, buffer: int = 0, no_overlap: bool = True
) -> Int[Tensor, "k 2"]:
    if buffer > 0:
        x = x[:, buffer:-buffer]
    indices = x.flatten().argsort(-1, descending=True)
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer

    if no_overlap:
        unique_indices = t.empty((0, 2), device=x.device).long()
        while len(unique_indices) < k:
            unique_indices = t.cat((unique_indices, t.tensor([[rows[0], cols[0]]], device=x.device)))
            is_overlapping_mask = (rows == rows[0]) & ((cols - cols[0]).abs() <= buffer)
            rows = rows[~is_overlapping_mask]
            cols = cols[~is_overlapping_mask]
        return unique_indices

    return t.stack((rows, cols), dim=1)[:k]


def index_with_buffer(
    x: Float[Tensor, "batch seq"], indices: Int[Tensor, "k 2"], buffer: int | None = None
) -> Float[Tensor, " k *buffer_x2_plus1"]:
    rows, cols = indices.unbind(dim=-1)
    if buffer is not None:
        rows = einops.repeat(rows, "k -> k buffer", buffer=buffer * 2 + 1)
        cols[cols < buffer] = buffer
        cols[cols > x.size(1) - buffer - 1] = x.size(1) - buffer - 1
        cols = einops.repeat(cols, "k -> k buffer", buffer=buffer * 2 + 1) + t.arange(
            -buffer, buffer + 1, device=x.device
        )
    return x[rows, cols]


def display_top_seqs(data: list[tuple[float, list[str], int]]):
    table = Table("Act", "Sequence", title="Max Activating Examples", show_lines=True)
    for act, str_toks, seq_pos in data:
        formatted_seq = (
            "".join([f"[b u green]{str_tok}[/]" if i == seq_pos else str_tok for i, str_tok in enumerate(str_toks)])
            .replace("�", "")
            .replace("\n", "↵")
        )
        table.add_row(f"{act:.3f}", repr(formatted_seq))
    rprint(table)


def fetch_max_activating_examples(
    model: HookedSAETransformer,
    transcoder: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 100,
    k: int = 10,
    buffer: int = 10,
    display: bool = False,
) -> list[tuple[float, list[str], int]]:
    data = []

    for _ in tqdm(range(total_batches)):
        tokens = act_store.get_batch_tokens()
        cache = run_with_cache_with_transcoder(model, [transcoder], tokens)
        acts = cache[f"{transcoder.cfg.metadata.hook_name}.hook_sae_acts_post"][..., latent_idx]

        k_largest_indices = get_k_largest_indices(acts, k=k, buffer=buffer)
        tokens_with_buffer = index_with_buffer(tokens, k_largest_indices, buffer=buffer)
        str_toks = [model.to_str_tokens(toks) for toks in tokens_with_buffer]
        top_acts = index_with_buffer(acts, k_largest_indices).tolist()
        data.extend(list(zip(top_acts, str_toks, [buffer] * len(str_toks))))

    data = sorted(data, key=lambda x: x[0], reverse=True)[:k]
    if display:
        display_top_seqs(data)
    return data

# %%
latent_idx = 1
neuronpedia_id = "gpt2-small/8-tres-dc"
url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
display(IFrame(url, width=800, height=600))

fetch_max_activating_examples(
    gpt2, gpt2_transcoder, gpt2_act_store, latent_idx=latent_idx, total_batches=200, display=True
)

# %%
