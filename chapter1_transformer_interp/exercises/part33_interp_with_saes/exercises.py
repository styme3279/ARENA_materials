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
_, cache = gpt2.run_with_cache_with_saes(prompt, saes=[gpt2_sae])

for name, param in cache.items():
    if "hook_sae" in name:
        print(f"{name:<43}: {tuple(param.shape)}")

# %%
# Get top activations on final token
_, cache = gpt2.run_with_cache_with_saes(
    prompt,
    saes=[gpt2_sae],
    stop_at_layer=_get_hook_layer(gpt2_sae) + 1,
)
sae_acts_post = cache[f"{gpt2_sae.cfg.metadata.hook_name}.hook_sae_acts_post"][0, -1, :]

# Plot line chart of latent activations
px.line(
    sae_acts_post.cpu().numpy(),
    title=f"Latent activations at the final token position ({sae_acts_post.nonzero().numel()} alive)",
    labels={"index": "Latent", "value": "Activation"},
    width=1000,
).update_layout(showlegend=False).show()

# Print the top 5 latents, and inspect their dashboards
for act, ind in zip(*sae_acts_post.topk(3)):
    print(f"Latent {ind} had activation {act:.2f}")
    display_dashboard(latent_idx=ind)

# %%
logits_no_saes, cache_no_saes = gpt2.run_with_cache(prompt)

gpt2_sae.use_error_term = False
logits_with_sae_recon, cache_with_sae_recon = gpt2.run_with_cache_with_saes(prompt, saes=[gpt2_sae])

gpt2_sae.use_error_term = True
logits_without_sae_recon, cache_without_sae_recon = gpt2.run_with_cache_with_saes(prompt, saes=[gpt2_sae])

# Both SAE caches contain the hook values
assert f"{gpt2_sae.cfg.metadata.hook_name}.hook_sae_acts_post" in cache_with_sae_recon
assert f"{gpt2_sae.cfg.metadata.hook_name}.hook_sae_acts_post" in cache_without_sae_recon

# But final output will be different, because we don't use SAE reconstructions when use_error_term
t.testing.assert_close(logits_no_saes, logits_without_sae_recon)
logit_diff_from_sae = (logits_no_saes - logits_with_sae_recon).abs().mean()
print(f"Average logit diff from using SAE reconstruction: {logit_diff_from_sae:.4f}")

# %%
gpt2_act_store = ActivationsStore.from_sae(
    model=gpt2,
    sae=gpt2_sae,
    dataset="NeelNanda/pile-10k",
    streaming=True,
    store_batch_size_prompts=16,
    n_batches_in_buffer=32,
    device=str(device),
)

# Example of how you can use this:
tokens = gpt2_act_store.get_batch_tokens()
assert tokens.shape == (gpt2_act_store.store_batch_size_prompts, gpt2_act_store.context_size)

# %%
display_dashboard(latent_idx=9)

# %%

def show_activation_histogram(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 200,
):
    """
    Displays the activation histogram for a particular latent, computed across `total_batches`
    batches from `act_store`.
    """


    sae_acts_post_hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
    all_positive_acts = []
    for batch in range(total_batches):
        tokens = act_store.get_batch_tokens()
        _, cache = model.run_with_cache_with_saes(
            tokens,
            saes=[sae],
            stop_at_layer=_get_hook_layer(sae) + 1,
        )

        # print(cache[sae_acts_post_hook_name].shape)
        all_activations = cache[sae_acts_post_hook_name][:, :, latent_idx]
        # print(all_activations.shape)
        all_activations = einops.rearrange(all_activations, "b seq -> (b seq)")
        # mask = all_activations >= 0.0

        for i in range(all_activations.shape[0]):
            act = all_activations[i].item()
            if act > 0.0:
                all_positive_acts.append(act)

    frac_active = len(all_positive_acts) / (
        total_batches * act_store.store_batch_size_prompts * act_store.context_size
    )

    px.histogram(
        all_positive_acts,
        nbins=50,
        title=f"ACTIVATIONS DENSITY {frac_active:.3%}",
        labels={"value": "Activation"},
        width=800,
        template="ggplot2",
        color_discrete_sequence=["darkorange"],
    ).update_layout(bargap=0.02, showlegend=False).show()


show_activation_histogram(gpt2, gpt2_sae, gpt2_act_store, latent_idx=9)

# %%
def get_k_largest_indices(x: Float[Tensor, "batch seq"], k: int, buffer: int = 0) -> Int[Tensor, "k 2"]:
    """
    The indices of the top k elements in the input tensor, i.e. output[i, :] is the (batch, seqpos)
    value of the i-th largest element in x.

    Won't choose any elements within `buffer` from the start or end of their sequence.
    """
    if buffer > 0:
        x = x[:, buffer:-buffer]
    indices = x.flatten().topk(k=k).indices
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer
    return t.stack((rows, cols), dim=1)


x = t.arange(40, device=device).reshape((2, 20))
x[0, 10] += 50  # 2nd highest value
x[0, 11] += 100  # highest value
x[1, 1] += 150  # not inside buffer (it's less than 3 from the start of the sequence)
top_indices = get_k_largest_indices(x, k=2, buffer=3)
assert top_indices.tolist() == [[0, 11], [0, 10]]


def index_with_buffer(
    x: Float[Tensor, "batch seq"], indices: Int[Tensor, "k 2"], buffer: int | None = None
) -> Float[Tensor, " k *buffer_x2_plus1"]:
    """
    Indexes into `x` with `indices` (which should have come from the `get_k_largest_indices`
    function), and takes a +-buffer range around each indexed element. If `indices` are less than
    `buffer` away from the start of a sequence then we just take the first `2*buffer+1` elems (same
    for at the end of a sequence).

    If `buffer` is None, then we don't add any buffer and just return the elements at the given indices.
    """
    rows, cols = indices.unbind(dim=-1)
    if buffer is not None:
        rows = einops.repeat(rows, "k -> k buffer", buffer=buffer * 2 + 1)
        cols[cols < buffer] = buffer
        cols[cols > x.size(1) - buffer - 1] = x.size(1) - buffer - 1
        cols = einops.repeat(cols, "k -> k buffer", buffer=buffer * 2 + 1) + t.arange(
            -buffer, buffer + 1, device=cols.device
        )
    return x[rows, cols]


x_top_values_with_context = index_with_buffer(x, top_indices, buffer=3)
assert x_top_values_with_context[0].tolist() == [
    8,
    9,
    10 + 50,
    11 + 100,
    12,
    13,
    14,
]  # highest value in the middle
assert x_top_values_with_context[1].tolist() == [
    7,
    8,
    9,
    10 + 50,
    11 + 100,
    12,
    13,
]  # 2nd highest value in the middle


def display_top_seqs(data: list[tuple[float, list[str], int]]):
    """
    Given a list of (activation: float, str_toks: list[str], seq_pos: int), displays a table of
    these sequences, with the relevant token highlighted.

    We also turn newlines into "\\n", and remove unknown tokens � (usually weird quotation marks)
    for readability.
    """
    table = Table("Act", "Sequence", title="Max Activating Examples", show_lines=True)
    for act, str_toks, seq_pos in data:
        formatted_seq = (
            "".join([f"[b u green]{str_tok}[/]" if i == seq_pos else str_tok for i, str_tok in enumerate(str_toks)])
            .replace("�", "")
            .replace("\n", "↵")
        )
        table.add_row(f"{act:.3f}", repr(formatted_seq))
    rprint(table)


example_data = [
    (0.5, [" one", " two", " three"], 0),
    (1.5, [" one", " two", " three"], 1),
    (2.5, [" one", " two", " three"], 2),
]
display_top_seqs(example_data)

# %%
def fetch_max_activating_examples(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 100,
    k: int = 10,
    buffer: int = 10,
) -> list[tuple[float, list[str], int]]:
    """
    Returns the max activating examples across a number of batches from the activations store.
    """
    sae_acts_post_hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"

    # Create list to store the top k activations for each batch. Once we're done,
    # we'll filter this to only contain the top k over all batches
    data = []

    for _ in tqdm(range(total_batches), desc="Computing activations for max activating examples"):
        tokens = act_store.get_batch_tokens()
        _, cache = model.run_with_cache_with_saes(
            tokens,
            saes=[sae],
            stop_at_layer=_get_hook_layer(sae) + 1,
            names_filter=[sae_acts_post_hook_name],
        )
        acts = cache[sae_acts_post_hook_name][..., latent_idx]

        # Get largest indices, get the corresponding max acts, and get the surrounding indices
        k_largest_indices = get_k_largest_indices(acts, k=k, buffer=buffer)
        tokens_with_buffer = index_with_buffer(tokens, k_largest_indices, buffer=buffer)
        str_toks = [model.to_str_tokens(toks) for toks in tokens_with_buffer]
        top_acts = index_with_buffer(acts, k_largest_indices).tolist()
        data.extend(list(zip(top_acts, str_toks, [buffer] * len(str_toks))))

    return sorted(data, key=lambda x: x[0], reverse=True)[:k]

# Fetch & display the results
buffer = 10
data = fetch_max_activating_examples(gpt2, gpt2_sae, gpt2_act_store, latent_idx=9, buffer=buffer, k=5)
display_top_seqs(data)

# Test one of the results, to see if it matches the expected output
first_seq_str_tokens = data[0][1]
assert first_seq_str_tokens[buffer] == " new"

# %%
data = fetch_max_activating_examples(gpt2, gpt2_sae, gpt2_act_store, latent_idx=16873, total_batches=200)
display_top_seqs(data)

# %%
def get_k_largest_indices(
    x: Float[Tensor, "batch seq"],
    k: int,
    buffer: int = 0,
    no_overlap: bool = True,
) -> Int[Tensor, "k 2"]:
    """
    Returns the tensor of (batch, seqpos) indices for each of the top k elements in the tensor x.

    Args:
        buffer:     We won't choose any elements within `buffer` from the start or end of their seq
                    (this helps if we want more context around the chosen tokens).
        no_overlap: If True, this ensures that no 2 top-activating tokens are in the same seq and
                    within `buffer` of each other.
    """
    assert buffer * 2 < x.size(1), "Buffer is too large for the sequence length"
    assert not no_overlap or k <= x.size(0), "Not enough sequences to have a different token in each sequence"

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


x = t.arange(40, device=device).reshape((2, 20))
x[0, 10] += 150  # highest value
x[0, 11] += 100  # 2nd highest value, but won't be chosen because of overlap
x[1, 10] += 50  # 3rd highest, will be chosen
top_indices = get_k_largest_indices(x, k=2, buffer=3)
assert top_indices.tolist() == [[0, 10], [1, 10]]


data = fetch_max_activating_examples(gpt2, gpt2_sae, gpt2_act_store, latent_idx=16873, total_batches=200)
display_top_seqs(data)

# %%
def show_top_logits(
    model: HookedSAETransformer,
    sae: SAE,
    latent_idx: int,
    k: int = 10,
) -> None:
    """
    Displays the top & bottom logits for a particular latent.
    """
    decoded_vector = sae.W_dec[latent_idx]
    logits = decoded_vector @ model.W_U

    pos_logits, pos_token_ids = t.topk(logits, k)

    pos_tokens = []
    for token_id in pos_token_ids:
        pos_tokens.append(model.tokenizer.decode(token_id))

    neg_logits, neg_token_ids = t.topk(logits, k, largest=False)

    neg_tokens = []
    for token_id in neg_token_ids:
        neg_tokens.append(model.tokenizer.decode(token_id))


    print(type(pos_tokens))

    print(
        tabulate(
            zip(map(repr, neg_tokens), neg_logits, map(repr, pos_tokens), pos_logits),
            headers=["Bottom tokens", "Value", "Top tokens", "Value"],
            tablefmt="simple_outline",
            stralign="right",
            numalign="left",
            floatfmt="+.3f",
        )
    )


show_top_logits(gpt2, gpt2_sae, latent_idx=9)
tests.test_show_top_logits(show_top_logits, gpt2, gpt2_sae)

# %%
def get_autointerp_df(sae_release="gpt2-small-res-jb", sae_id="blocks.7.hook_resid_pre") -> pd.DataFrame:
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]  # e.g. "gpt2-small/7-res-jb"

    # Neuronpedia publishes its explanations as gzipped JSONL files in a public S3 bucket (see
    # https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/ - there are several files per
    # SAE, so we list all the files for this SAE then download & concatenate them)
    bucket_url = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
    params = {"list-type": "2", "prefix": f"v1/{neuronpedia_id}/explanations/"}
    listing = requests.get(bucket_url, params=params)
    listing.raise_for_status()
    keys = re.findall(r"<Key>(.*?)</Key>", listing.text)
    assert len(keys) > 0, f"No explanations found on Neuronpedia for {neuronpedia_id!r}"

    columns = ["modelId", "layer", "index", "description", "explanationModelName", "typeName"]
    df = pd.concat(
        [
            pd.read_json(f"{bucket_url}/{key}", lines=True, compression="gzip", dtype={"index": int})[columns]
            for key in tqdm(keys, desc=f"Downloading explanations for {neuronpedia_id}")
        ],
        ignore_index=True,
    )
    return df.sort_values("index", kind="stable", ignore_index=True)


explanations_df = get_autointerp_df()
explanations_df.head()

# %%
def create_prompt(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 100,
    k: int = 15,
    buffer: int = 10,
) -> dict[Literal["system", "user", "assistant"], str]:
    """
    Returns the system, user & assistant prompts for autointerp.
    """
    data = fetch_max_activating_examples(model, sae, act_store, latent_idx, total_batches, k, buffer)
    str_formatted_examples = "\n".join(
        f"{i + 1}. {''.join(f'<<{tok}>>' if j == buffer else tok for j, tok in enumerate(seq[1]))}"
        for i, seq in enumerate(data)
    )

    return {
        "system": "We're studying neurons in a neural network. Each neuron activates on some particular word or concept in a short document. The activating words in each document are indicated with << ... >>. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Try to be specific in your explanations, although don't be so specific that you exclude some of the examples from matching your explanation. Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words.",
        "user": f"""The activating documents are given below:\n\n{str_formatted_examples}""",
        "assistant": "this neuron fires on",
    }

# %%
def get_autointerp_explanation(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 100,
    k: int = 15,
    buffer: int = 10,
    n_completions: int = 5,
) -> list[str]:
    """
    Queries OpenAI's API using prompts returned from `create_prompt`, and returns a list of the
    completions.
    """
    client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

    prompts = create_prompt(model, sae, act_store, latent_idx, total_batches, k, buffer)

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
            {"role": "assistant", "content": prompts["assistant"]},
        ],
        n=n_completions,
        max_tokens=50,
        stream=False,
    )
    return [choice.message.content for choice in result.choices]

API_KEY = os.environ.get("OPENROUTER_API_KEY", None)

if API_KEY is not None:
    completions = get_autointerp_explanation(gpt2, gpt2_sae, gpt2_act_store, latent_idx=9, n_completions=5)
    for i, completion in enumerate(completions):
        print(f"Completion {i + 1}: {completion!r}")
else:
    print("No API key found, not running the autointerp code.")

# %%
attn_saes = {
    layer: SAE.from_pretrained(
        "gpt2-small-hook-z-kk",
        f"blocks.{layer}.hook_z",
        device=str(device),
    )
    for layer in range(gpt2.cfg.n_layers)
}

layer = 9

display_dashboard(
    sae_release="gpt2-small-hook-z-kk",
    sae_id=f"blocks.{layer}.hook_z",
    latent_idx=2,  # or you can try `random.randint(0, attn_saes[layer].cfg.d_sae)`
)

# %%
