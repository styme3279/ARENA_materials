
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
# print(f"gpt2_small.cfg: {gpt2_small.cfg}")

print(f"gpt2_small.cfg.n_layers: {gpt2_small.cfg.n_layers}")
print(f"gpt2_small.cfg.n_heads: {gpt2_small.cfg.n_heads}")
print(f"gpt2_small.cfg.n_ctx: {gpt2_small.cfg.n_ctx}")



# %%
model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model out, let's find the loss on this paragraph!"""

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)




# %%
print(f"GPT W_Q shape: {gpt2_small.W_Q.shape}")
print(f"GPT W_in shape: {gpt2_small.W_in.shape}")
print(f"GPT W_out shape: {gpt2_small.W_out.shape}")

# %%
print(gpt2_small.to_str_tokens("gpt2"))
print(gpt2_small.to_str_tokens(["gpt2", "gpt2"]))
print(gpt2_small.to_tokens("gpt2"))
print(gpt2_small.to_string([50256, 70, 457, 17]))

# %%
logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]

# YOUR CODE HERE - get the model's prediction on the text
print(f"prediction text: {gpt2_small.to_string(prediction)}")
print(f"prediction: {prediction}")
print(f"prediction shape: {prediction.shape}")
true_tokens = gpt2_small.to_tokens(model_description_text)[:,1:]
print(f"true_tokens: {true_tokens}")
print(f"true_tokens: {true_tokens.shape}")
accuracy = (true_tokens == prediction).sum()
print(f"accuracy: {accuracy / true_tokens.shape[1]} %")




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

# YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the
# steps of the attention calculation (dot product, masking, scaling, softmax)
print(F"layer0_pattern_from_cache shape: {layer0_pattern_from_cache.shape}")

seq, nhead, headsize = q.shape
q = gpt2_cache["q", 0]
k = gpt2_cache["k", 0]
print(F"q shape: {q.shape}")
print(F"k shape: {k.shape}")
attn_raw = einops.einsum(q, k, "s_len_q n h, s_len_k n h -> n s_len_q s_len_k")
print(f"attn_raw shape: {attn_raw.shape}")

mask = t.triu(t.ones((seq, seq), dtype=t.bool), diagonal=1).to(device)
attn_raw.masked_fill_(mask, -1e9)
layer0_pattern_from_q_and_k = (attn_raw / headsize**0.5).softmax(-1)



t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")

# %%
