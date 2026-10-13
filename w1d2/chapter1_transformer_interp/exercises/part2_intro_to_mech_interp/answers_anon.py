
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

from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)



# %%
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

# %%
attention_pattern = cache["pattern", 0]
str_tokens = model.to_str_tokens(text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)
attention_pattern = cache["pattern", 1]
str_tokens = model.to_str_tokens(text)
display(
    cv.attention.attention_patterns(
        tokens=str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L1H{i}" for i in range(12)],
    )
)
# %%
def current_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """
    # print(f"cfg.n_layers: {cfg.n_layers}")
    cur_attn_list = []
    threshold = 25
    for layer in range(cfg.n_layers):
        attn_all_heads = cache["pattern", layer]
        # print(f"attn_all_heads: {attn_all_heads.shape}")
        attn_diag = t.eye(attn_all_heads.shape[-1], device=device) * attn_all_heads
        # print(f"attn_all_heads[0]: {attn_all_heads[0]}")
        # print(f"attn_diag[0]: {attn_diag[0]}")
        total_attn = attn_diag.sum(-1).sum(-1)
        # print(f"total attn: {total_attn}")
        for i, tot_attn_h in enumerate(total_attn):
            if tot_attn_h > threshold:
                cur_attn_list.append(f"{layer}.{i}")

    return cur_attn_list


def prev_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """
        # print(f"cfg.n_layers: {cfg.n_layers}")
    prev_attn_list = []
    threshold = 25
    for layer in range(cfg.n_layers):
        attn_all_heads = cache["pattern", layer]
        # print(f"attn_all_heads: {attn_all_heads.shape}")
        full_mat = t.ones((attn_all_heads.shape[-1], attn_all_heads.shape[-1]))
        diag_min_1 = t.tril(full_mat, diagonal=-1) - t.tril(full_mat, diagonal=-2)
        # print(f"diag_min_1: {diag_min_1}")

        attn_diag = diag_min_1.to(device) * attn_all_heads
        # print(f"attn_all_heads[0]: {attn_all_heads[0]}")
        # print(f"attn_diag[0]: {attn_diag[0]}")
        total_attn = attn_diag.sum(-1).sum(-1)
        # print(f"total attn: {total_attn}")
        for i, tot_attn_h in enumerate(total_attn):
            if tot_attn_h > threshold:
                prev_attn_list.append(f"{layer}.{i}")
    return prev_attn_list

def first_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """
    first_attn_list = []
    threshold = 30
    for layer in range(cfg.n_layers):
        attn_all_heads = cache["pattern", layer]
        # print(f"attn_all_heads: {attn_all_heads.shape}")
        full_mat = t.zeros((attn_all_heads.shape[-1], attn_all_heads.shape[-1]))
        full_mat[:, 0] = 1.0

        attn_diag = full_mat.to(device) * attn_all_heads
        total_attn = attn_diag.sum(-1).sum(-1)
        # print(f"total attn: {total_attn}")
        for i, tot_attn_h in enumerate(total_attn):
            if tot_attn_h > threshold:
                first_attn_list.append(f"{layer}.{i}")

    return first_attn_list


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))

# %%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    t.manual_seed(0)  # for reproducibility
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch_size, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    print(f"rep_tokens: {rep_tokens}")
    return rep_tokens


def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens,
    logits, cache). This function should use the `generate_repeated_tokens` function above.

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    rep_tokens = generate_repeated_tokens(model, seq_len, batch_size)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache



def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs


seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len)

# %%

# seq_len = 50



def induction_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of (-seq_len+1)-offset elements
            seq_len = (attention_pattern.shape[-1] - 1) // 2

            score = attention_pattern.diagonal(-seq_len + 1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads


print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %%
seq_len = 50
batch_size = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU,
# which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)


def induction_score_hook(pattern: Float[Tensor, "batch head_index dest_pos source_pos"], hook: HookPoint):
    """
    Calculates the induction score, and stores it in the [layer, head] position of the
    `induction_score_store` tensor.
    """
    layer = hook.layer()

    batch_size, num_heads, num_s_q, num_s_k = pattern.shape
    seq_len = (num_s_q - 1) // 2

    attn_pattn_mean = pattern.diagonal(-seq_len+1, dim1=2, dim2=3).mean((0, -1))
    induction_score_store[layer] = attn_pattn_mean

    return pattern


# We make a boolean filter on activation names, that's true only on attention pattern names
pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%

def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(cv.attention.attention_patterns(tokens=gpt2_small.to_str_tokens(rep_tokens[0]), attention=pattern.mean(0)))

# YOUR CODE HERE - find induction heads in gpt2_small

seq_len = 50
batch_size = 10
rep_tokens_batch = generate_repeated_tokens(gpt2_small, seq_len, batch_size)


induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)

gpt2_small.run_with_hooks(
    rep_tokens_batch,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)
# Plot the induction scores for each head in each layer
imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=500,
    height=500,
)

num_layers = [5, 7, 10]
fwd_hooks = [
    (utils.get_act_name("pattern", l), visualize_pattern_hook) for l in num_layers
]


gpt2_small.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    # fwd_hooks=[(pattern_hook_names_filter, visualize_pattern_hook)],
    fwd_hooks=fwd_hooks,
)


# %%

def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"],
) -> Float[Tensor, "seq-1 n_components"]:
    """
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dims)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dims)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    """
    W_U_correct_tokens = W_U[:, tokens[1:]]

    direct = einops.einsum(W_U_correct_tokens, embed[:-1], "emb seq, seq emb -> seq")
    l1_attributions = einops.einsum(W_U_correct_tokens, l1_results[:-1], "emb seq, seq nhead emb -> seq nhead")
    l2_attributions = einops.einsum(W_U_correct_tokens, l2_results[:-1], "emb seq, seq nhead emb -> seq nhead")
    return t.concat([direct.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)


text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")

embed = cache["embed"]
l1_results = cache["result", 0]
l2_results = cache["result", 1]
logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens.squeeze())

plot_logit_attribution(model, logit_attr, tokens, title="Logit attribution (demo prompt)")

# %%
# YOUR CODE HERE - plot logit attribution for the induction sequence (i.e. using `rep_tokens` and
# `rep_cache`), and interpret the results.

seq_len = 50

embed = rep_cache["embed"]
l1_results = rep_cache["result", 0]
l2_results = rep_cache["result", 1]

logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, rep_tokens.squeeze())
plot_logit_attribution(model, logit_attr, rep_tokens.squeeze(), title="Logit attribution (random induction prompt)")


# %%
def head_zero_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:

    # layer = hook.layer()
    z[:,:,head_index_to_ablate,:] = 0
    
    return


def get_ablation_scores(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    ablation_function: Callable = head_zero_ablation_hook,
) -> Float[Tensor, "n_layers n_heads"]:
    """
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss
    from ablating the output of each head.
    """
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = model(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            hook_func = functools.partial(head_zero_ablation_hook, head_index_to_ablate = head)
            
            logits_ablated = model.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=[(utils.get_act_name("z", layer), hook_func)],
            )
            loss = -get_log_probs(logits_ablated, tokens)[:, -(seq_len - 1) :].mean()
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


ablation_scores = get_ablation_scores(model, rep_tokens)
tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)


imshow(
    ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Ablating Heads",
    text_auto=".2f",
    width=900,
    height=350,
)
# %%

def head_mean_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    
    # over batch dimension
    z[:,:,head_index_to_ablate,:] = z[:,:,head_index_to_ablate,:].mean(0)
    
    return


rep_tokens_batch = run_and_cache_model_repeated_tokens(model, seq_len=50, batch_size=10)[0]
mean_ablation_scores = get_ablation_scores(model, rep_tokens_batch, ablation_function=head_mean_ablation_hook)

# imshow(
#     mean_ablation_scores,
#     labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
#     title="Loss Difference After Ablating Heads",
#     text_auto=".2f",
#     width=900,
#     height=350,
# )


# # %%

A = t.randn(5, 2)
B = t.randn(2, 5)
AB = A @ B
AB_factor = FactoredMatrix(A, B)
print("Norms:")
print(AB.norm())
print(AB_factor.norm())

print(f"Right dim: {AB_factor.rdim}, Left dim: {AB_factor.ldim}, Hidden dim: {AB_factor.mdim}")

print("Eigenvalues:")
print(t.linalg.eig(AB).eigenvalues)
print(AB_factor.eigenvalues)

print("\nSingular Values:")
print(t.linalg.svd(AB).S)
print(AB_factor.S)

print("\nFull SVD:")
print(AB_factor.svd())


# %%
C = t.randn(5, 300)
ABC = AB @ C
ABC_factor = AB_factor @ C

print(f"Unfactored: shape={ABC.shape}, norm={ABC.norm()}")
print(f"Factored: shape={ABC_factor.shape}, norm={ABC_factor.norm()}")
print(f"\nRight dim: {ABC_factor.rdim}, Left dim: {ABC_factor.ldim}, Hidden dim: {ABC_factor.mdim}")

AB_unfactored = AB_factor.AB
t.testing.assert_close(AB_unfactored, AB)

# %%
head_index = 4
layer = 1

# W_E W_OV W_U
# W_OV = W_V W_O

W_E = model.W_E
W_U = model.W_U
W_V_14 = model.W_V[1,4]
W_O_14 = model.W_O[1,4]

full_OV_circuit = W_E @ FactoredMatrix(W_V_14, W_O_14) @ W_U

tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)

# %%
indices = t.randint(0, model.cfg.d_vocab, (200,))
full_OV_circuit_sample = full_OV_circuit[indices, indices].AB

imshow(
    full_OV_circuit_sample,
    labels={"x": "Logits on output token", "y": "Input token"},
    title="Full OV circuit for copying head",
    width=700,
    height=600,
)


# %%
def top_1_acc(full_OV_circuit: FactoredMatrix, batch_size: int = 1000) -> float:
    """
    Return the fraction of the time that the maximum value is on the circuit diagonal.
    """
    total = 0
    for indices in t.split(t.arange(full_OV_circuit.shape[0], device=device), batch_size):
        AB_slice = full_OV_circuit[indices].AB
        total += (t.argmax(AB_slice, dim=1) == indices).float().sum().item()

    return total / full_OV_circuit.shape[0]


print(f"Fraction of time that the best logit is on diagonal: {top_1_acc(full_OV_circuit):.4f}")

# %%

W_E = model.W_E
W_U = model.W_U
W_V_14 = model.W_V[1,4]
W_O_14 = model.W_O[1,4]

W_V_110 = model.W_V[1,10]
W_O_110 = model.W_O[1,10]

OV_both = FactoredMatrix(t.concat((W_V_14, W_V_110), dim=1), t.concat((W_O_14, W_O_110), dim=0))

full_OV_circuit = W_E @ OV_both @ W_U
print(f"Fraction of time that the best logit is on diagonal: {top_1_acc(full_OV_circuit):.4f}")


# %%
layer = 0
head_index = 7

# Compute full QK matrix (for positional embeddings)
W_pos = model.W_pos
W_QK = model.W_Q[layer, head_index] @ model.W_K[layer, head_index].T
pos_by_pos_scores = W_pos @ W_QK @ W_pos.T

# Mask, scale and softmax the scores
mask = t.tril(t.ones_like(pos_by_pos_scores)).bool()
pos_by_pos_pattern = t.where(mask, pos_by_pos_scores / model.cfg.d_head**0.5, -1.0e6).softmax(-1)

# Plot the results
print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")
imshow(
    utils.to_numpy(pos_by_pos_pattern[:200, :200]),
    labels={"x": "Key", "y": "Query"},
    title="Attention patterns for prev-token QK circuit, first 100 indices",
    width=700,
    height=600,
)



# %%

def decompose_qk_input(cache: ActivationCache) -> Float[Tensor, "n_heads+2 posn d_model"]:
    """
    Retrieves all the input tensors to the first attention layer, and concatenates them along the
    0th dim.

    The [i, :, :]th element is y_i (from notation above). The sum of these tensors along the 0th
    dim should be the input to the first attention layer.
    """
    y0 = cache["embed"].unsqueeze(0)  # shape (1, seq, d_model)
    y1 = cache["pos_embed"].unsqueeze(0)  # shape (1, seq, d_model)
    y_rest = cache["result", 0].transpose(0, 1)  # shape (12, seq, d_model)

    return t.concat([y0, y1, y_rest], dim=0)


def decompose_q(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of query vectors for each decomposed QK input.

    The [i, :, :]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values).
    """
    W_Q = model.W_Q[1, ind_head_index]

    return einops.einsum(decomposed_qk_input, W_Q, "n seq d_model, d_model d_head -> n seq d_head")


def decompose_k(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of key vectors for each decomposed QK input.

    The [i, :, :]th element is y_i @ W_K (so the sum along axis 0 is just the k-values)
    """
    W_K = model.W_K[1, ind_head_index]

    return einops.einsum(decomposed_qk_input, W_K, "n seq d_model, d_model d_head -> n seq d_head")


# Recompute rep tokens/logits/cache, if we haven't already
seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
rep_cache.remove_batch_dim()

ind_head_index = 10

# First we get decomposed q and k input, and check they're what we expect
decomposed_qk_input = decompose_qk_input(rep_cache)
decomposed_q = decompose_q(decomposed_qk_input, ind_head_index, model)
decomposed_k = decompose_k(decomposed_qk_input, ind_head_index, model)
t.testing.assert_close(
    decomposed_qk_input.sum(0),
    rep_cache["resid_pre", 1] + rep_cache["pos_embed"],
    rtol=0.01,
    atol=1e-05,
)
t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)

# Second, we plot our results
component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
    imshow(
        utils.to_numpy(decomposed_input.pow(2).sum([-1])),
        labels={"x": "Position", "y": "Component"},
        title=f"Norms of components of {name}",
        y=component_labels,
        width=800,
        height=400,
    )

# %%
def decompose_attn_scores(
    decomposed_q: Float[Tensor, "q_comp q_pos d_head"],
    decomposed_k: Float[Tensor, "k_comp k_pos d_head"],
    model: HookedTransformer,
) -> Float[Tensor, "q_comp k_comp q_pos k_pos"]:
    """
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]

    The [i, j, 0, 0]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the
    attention scores)
    """
    return einops.einsum(decomposed_q, decomposed_k, "q_comp q_pos d_head, k_comp k_pos d_head -> q_comp k_comp q_pos k_pos") / (model.cfg.d_head**.5)


tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k, model)

# First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7), you can replace this
# with any other pair and see that the values are generally much smaller, i.e. this pair dominates the attention score
# calculation
decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k, model)

q_label = "Embed"
k_label = "0.7"
decomposed_scores_from_pair = decomposed_scores[component_labels.index(q_label), component_labels.index(k_label)]

imshow(
    utils.to_numpy(t.tril(decomposed_scores_from_pair)),
    title=f"Attention score contributions from query = {q_label}, key = {k_label}<br>(by query & key sequence positions)",
    width=700,
)


# Second plot: std dev over query and key positions, shown by component. This shows us that the other pairs of
# (query_component, key_component) are much less important, without us having to look at each one individually like we
# did in the first plot!
decomposed_stds = einops.reduce(
    decomposed_scores, "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", t.std
)
imshow(
    utils.to_numpy(decomposed_stds),
    labels={"x": "Key Component", "y": "Query Component"},
    title="Std dev of attn score contributions across sequence positions<br>(by query & key comp)",
    x=component_labels,
    y=component_labels,
    width=700,
)

# %%
