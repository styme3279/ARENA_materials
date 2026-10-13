# %%

import os
import sys
from pathlib import Path

from importlib.metadata import packages_distributions

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter1_transformer_interp"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
if "transformer_lens" not in packages_distributions():
    %pip install "transformer_lens>=3.9,<4" einops git+https://github.com/ARENA-education/eindex.git jaxtyping git+https://github.com/ARENA-education/CircuitsVis.git@arena-tl4.1#subdirectory=python

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
        !rmdir {root}/ARENA_materials-{branch}


if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises") 
# %%
import functools
import sys
from pathlib import Path
from typing import Callable

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
from eindex import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

import part2_intro_to_mech_interp.tests as tests
from plotly_utils import (
    hist,
    imshow,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"
# %%
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
# %%
print(gpt2_small.cfg)
# %%
model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model out, let's find the loss on this paragraph!"""

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)
# %%
print(gpt2_small.to_str_tokens("gpt2"))
print(gpt2_small.to_str_tokens(["gpt2", "gpt2"]))
print(gpt2_small.to_tokens("gpt2"))
print(gpt2_small.to_string([50256, 70, 457, 17]))
# %%
logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]

# YOUR CODE HERE - get the model's prediction on the text
true_tokens = gpt2_small.to_tokens(model_description_text, prepend_bos=False)


true_mask = t.where(prediction == true_tokens, 1, 0)
num_correct, correct_indices = true_mask.sum(), true_mask.nonzero()

print(f"correct tokens = {list(zip(gpt2_small.to_str_tokens(prediction[correct_indices[:, 1]]), correct_indices[:, 1].tolist()))}")
# %%
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_logits), type(gpt2_cache))
# %%
attn_patterns_from_shorthand = gpt2_cache["pattern", 0]
attn_patterns_from_full_name = gpt2_cache["blocks.0.attn.hook_pattern"]

t.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)
# %%
layer0_pattern_from_cache = gpt2_cache["pattern", 0]

layer0_q_from_cache, layer0_k_from_cache = gpt2_cache["q", 0], gpt2_cache["k", 0]

# attn_scores = layer0_q_from_cache.transpose(0, 1) @ layer0_k_from_cache.transpose(0, 1).transpose(-1, -2)  # (12, 33, 33)
"""
q # (33, 12, 64)
k # (33, 12, 64)

q # (12, 33, 64)
k # (12, 33, 64)


q # (12, 33, 64)
k # (12, 64, 33)

q @ k --> (12, 33, 33)



"""

attn_scores = einops.einsum(layer0_q_from_cache, layer0_k_from_cache, "seqQ head_idx d_head, seqK head_idx d_head -> head_idx seqQ seqK") # dot product
attn_scores *= 1 / gpt2_small.cfg.d_head ** 0.5 # scale
attn_scores = attn_scores.masked_fill(t.tril(attn_scores) == 0, t.tensor(float("-inf"), dtype=t.float32, device=device))

layer0_pattern_from_q_and_k = t.softmax(attn_scores, dim=-1)

# YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the
# steps of the attention calculation (dot product, masking, scaling, softmax)
t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")
# %%
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=gpt2_str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)
# %%
neuron_activations_for_all_layers = t.stack([
    gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)
], dim=1)
# shape = (seq_pos, layers, neurons)

cv.activations.text_neuron_activations(
    tokens=gpt2_str_tokens,
    activations=neuron_activations_for_all_layers
)
# %%
neuron_activations_for_all_layers_rearranged = utils.to_numpy(einops.rearrange(neuron_activations_for_all_layers, "seq layers neurons -> 1 layers seq neurons"))

cv.topk_tokens.topk_tokens(
    # Some weird indexing required here ¯\_(ツ)_/¯
    tokens=[gpt2_str_tokens],
    activations=neuron_activations_for_all_layers_rearranged,
    max_k=7,
    first_dimension_name="Layer",
    third_dimension_name="Neuron",
    first_dimension_labels=list(range(12))
)


# %%

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,  # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer",
)

# %%
from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
# %%


model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)


# %%
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)
# %%

attention_pattern = cache["pattern", 0]
print(attention_pattern.shape)
hooked_transformer_str_tokens = model.to_str_tokens(text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=hooked_transformer_str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)

attention_pattern = cache["pattern", 1]
print(attention_pattern.shape)
hooked_transformer_str_tokens = model.to_str_tokens(text)


print("Layer 1 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=hooked_transformer_str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L1H{i}" for i in range(12)],
    )
)


# %%
def current_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """
    head_idx, cache["pattern", 0].size())
    raise NotImplementedError()


def prev_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """
    raise NotImplementedError()


def first_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """
    raise NotImplementedError()


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))
# %%
