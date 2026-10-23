"""Utilities for the J-lens / J-space exercises (ARENA 1.3.5).

Everything here is plumbing that isn't the point of the exercises: loading the model and pre-fitted lenses,
recording the residual stream, fp32 unembedding, token bookkeeping and pretty-printing.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path(__file__).parent / "data"

MODEL_NAME = "Qwen/Qwen3.5-9B"

# Pre-fitted lenses for Qwen3.5-9B. Each file holds {"J": {layer: [d, d] tensor}, ...}.
#   - J / R: Camila Blank's matched J-lens and R-lens pair (25 Pile prompts x 128 tokens, skip_first=4,
#     target = penultimate block 30). The R-lens is the same fit with LRP stop-gradients.
#   - Jpp: Ayonrinde & Lindsey's J++ lens (64 WikiText prompts, E=8 experts mixed into one map, target = final
#     block 31), converted to fp16 by Neuronpedia.
LENS_FILES = {
    "J": ("camilablank/workspace-lenses", "qwen3.5-9b/j-lens/lens.pt"),
    "R": ("camilablank/workspace-lenses", "qwen3.5-9b/r-lens/lens.pt"),
    "Jpp": ("neuronpedia/jacobian-lens", "qwen3.5-9b/jpp/Salesforce-wikitext/Qwen3.5-9B_jpp_lens.pt"),
}

# Artifacts we fitted for these exercises (expert Jacobians for the J++ section, Gemma 4 lenses, etc.)
ARENA_REPO = "styme3279/arena-jspace-lenses"


# ============================================================
# Model loading and access
# ============================================================


def load_model(model_name: str = MODEL_NAME, device: str = "cuda", dtype=torch.bfloat16):
    """Loads the HF model (text-only causal LM) and tokenizer, with gradients disabled on all parameters."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def get_text_model(model):
    """Returns the bare decoder (the module holding .layers, .norm, .embed_tokens)."""
    for path in ["model.language_model", "model", "language_model"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "layers") and hasattr(obj, "norm"):
            return obj
    raise ValueError("Couldn't find the text decoder")


def get_layers(model) -> torch.nn.ModuleList:
    return get_text_model(model).layers


def final_norm_scale(model) -> Tensor:
    """The elementwise scale g applied by the final RMSNorm (Qwen3.5 / Gemma-3-style norms use 1 + w)."""
    norm = get_text_model(model).norm
    w = norm.weight.float()
    zero_centred = "Qwen3_5" in type(norm).__name__ or "Gemma3" in type(norm).__name__
    return (1.0 + w) if zero_centred else w


class Unembed:
    """fp32 version of the model's readout: final RMSNorm (with scale) followed by the unembedding.

    We deliberately skip any final logit softcap and do everything in fp32. The softcap is monotone so it never
    changes rankings, but in bf16 many logits saturate to exactly the same value, which creates ties and inflates
    rank-based metrics.
    """

    def __init__(self, model):
        self.W_U = model.lm_head.weight.float()  # [d_vocab, d_model]
        self.g = final_norm_scale(model)  # [d_model]
        self.eps = getattr(get_text_model(model).norm, "eps", 1e-6)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g
        return x @ self.W_U.T

    def token_direction(self, token_id: int) -> Tensor:
        """u_tau = g * W_U[tau]: the (unnormalised) logit-lens direction for a token."""
        return self.W_U[token_id] * self.g


# ============================================================
# Residual stream recording and hooks
# ============================================================


def _out_tensor(output):
    return output if torch.is_tensor(output) else output[0]


@contextmanager
def hooks(model, fns: dict[int, Callable]):
    """Registers forward hooks {layer: fn(module, input, output)} on the decoder blocks for the duration."""
    layers = get_layers(model)
    handles = [layers[l].register_forward_hook(f) for l, f in fns.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def get_resid(model, input_ids: Tensor, layers: Iterable[int] | None = None) -> Tensor:
    """Returns the residual stream after each block: [n_layers, seq, d_model] (fp32), for a batch of size 1."""
    n = len(get_layers(model))
    layers = list(range(n)) if layers is None else list(layers)
    acts = {}

    def rec(l):
        def f(m, i, o):
            acts[l] = _out_tensor(o)[0].float()

        return f

    with hooks(model, {l: rec(l) for l in layers}):
        get_text_model(model)(input_ids=input_ids, use_cache=False)
    return torch.stack([acts[l] for l in layers])


@torch.no_grad()
def next_token_logits(model, input_ids: Tensor, fns: dict[int, Callable] | None = None) -> Tensor:
    """Final-position logits (fp32), optionally with forward hooks applied."""
    with hooks(model, fns or {}):
        return model(input_ids=input_ids, use_cache=False).logits[0, -1].float()


# ============================================================
# Lenses
# ============================================================


def load_lens(name: str, device: str = "cuda") -> dict[int, Tensor]:
    """Loads a pre-fitted lens as {layer: J_l} (fp32 on device). `name` is a key of LENS_FILES, or (repo, filename)."""
    repo, filename = LENS_FILES[name] if isinstance(name, str) else name
    path = hf_hub_download(repo, filename)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return {int(l): J.to(device).float() for l, J in ckpt["J"].items()}


def load_jpp_experts(repo: str = ARENA_REPO, filename: str = "qwen3.5-9b/jpp_experts.pt", min_count: int = 50, device="cuda"):
    """Loads pre-computed J++ expert Jacobians (LRP backward, final-block target) for a few layers.

    Returns a dict with:
        experts:  {layer: [E, d, d]} expert Jacobians (experts with < min_count fit positions fall back to the pooled one)
        J_pooled: {layer: [d, d]} the pooled LRP Jacobian on the same data
        counts:   {layer: [E]} number of fit positions per expert
        route:    function (layer, h [seq, d]) -> cluster ids [seq]
    """
    ckpt = torch.load(hf_hub_download(repo, filename), map_location="cpu", weights_only=True)
    out = {"experts": {}, "J_pooled": {}, "counts": {}, "n_prompts": ckpt["n_prompts"]}
    router = {}
    for l, JE_sum in ckpt["JE_sum"].items():
        counts = ckpt["counts"][l].float()
        pooled = ckpt["J_pooled"][l].to(device).float()
        JE = JE_sum.to(device).float() / counts.to(device).clamp_min(1)[:, None, None]
        JE[counts.to(device) < min_count] = pooled
        out["experts"][l], out["J_pooled"][l], out["counts"][l] = JE, pooled, counts
        router[l] = [t.to(device) for t in ckpt["router"][l]]

    def route(layer: int, h: Tensor) -> Tensor:
        mu, P, C = router[layer]
        x = h.float() / h.float().norm(dim=-1, keepdim=True)
        return torch.cdist((x - mu) @ P, C).argmin(-1)

    out["route"] = route
    return out


# ============================================================
# Tokens
# ============================================================


def decode_vocab(tokenizer) -> list[str]:
    return [tokenizer.decode([i]) for i in range(len(tokenizer))]


def nonsemantic_mask(vocab: list[str], d_vocab: int, device="cuda") -> Tensor:
    """True for tokens whose string contains no letter or digit (whitespace, punctuation, padding rows).
    This is J++'s "Readout Filtering"."""
    mask = torch.ones(d_vocab, dtype=torch.bool)
    for i, s in enumerate(vocab[:d_vocab]):
        mask[i] = not re.search(r"\w", s)
    return mask.to(device)


def variant_ids(tokenizer, word: str) -> list[int]:
    """First-token ids of common surface forms of a word (" word", "Word", ...), so a word counts as present in a
    readout if any of its forms is."""
    out = set()
    for v in {word, " " + word, word.capitalize(), " " + word.capitalize(), word.lower(), " " + word.lower()}:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if ids:
            out.add(ids[0])
    return sorted(out)


def first_token_id(tokenizer, word: str, space: bool = True) -> int:
    return tokenizer.encode((" " if space else "") + word, add_special_tokens=False)[0]


def token_matches(vocab: list[str], token_id: int, word: str) -> bool:
    """Is `token_id` a plausible first token of `word`? (case-insensitive prefix match, ignoring whitespace)"""
    s = vocab[token_id].strip().lower()
    return len(s) > 0 and word.lower().startswith(s) and (len(s) >= 2 or len(word) == 1)


# ============================================================
# Data
# ============================================================


def load_json(name: str):
    """Loads one of the prompt sets (from anthropics/jacobian-lens's data/ folder, mirrored on our HF repo). Uses a
    local copy in data/ if there is one."""
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        path = hf_hub_download(ARENA_REPO, f"data/prompts/{name}.json")
    return json.load(open(path))


def load_lens_eval(task: str) -> list[dict]:
    """One of multihop, multilingual, typo, association, poetry, order-ops."""
    return load_json(f"lens-eval-{task}")["items"]


def load_pile_prompts(n: int, skip: int = 0) -> list[str]:
    """Documents from NeelNanda/pile-10k (the corpus used to fit the reference J-lens and R-lens)."""
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    return [ds[i]["text"] for i in range(skip, skip + n)]


def load_wikitext_prompts(n: int, min_chars: int = 600, seed_skip: int = 0) -> list[str]:
    """First n WikiText-103 records with at least `min_chars` characters (the recipe the J++ authors used)."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    out, skipped = [], 0
    for row in ds:
        text = row["text"].strip()
        if len(text) >= min_chars and not text.startswith("="):
            if skipped < seed_skip:
                skipped += 1
                continue
            out.append(text)
            if len(out) == n:
                break
    return out


# ============================================================
# Display
# ============================================================


def readout_table(rows: dict[str, list[str]], k: int | None = None) -> pd.DataFrame:
    """{row_label: [tok1, tok2, ...]} -> DataFrame with visible whitespace."""

    def clean(s):
        return s.replace("\n", "↵").replace(" ", "␣")

    k = k or max(len(v) for v in rows.values())
    return pd.DataFrame({r: [clean(t) for t in v[:k]] for r, v in rows.items()}).T
