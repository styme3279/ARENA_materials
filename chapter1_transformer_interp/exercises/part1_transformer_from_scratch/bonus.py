
# %%
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import datasets
import einops
import numpy as np
import torch as t
import torch.nn as nn
import wandb
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers import GPT2TokenizerFast
from time import time

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part1_transformer_from_scratch"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_transformer_from_scratch.solutions as solutions
import part1_transformer_from_scratch.tests as tests

MAIN = __name__ == "__main__"

# %%
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
)


# %%
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


cfg = Config()
print(cfg)


# %%
# Define a type for a single layer's cache entry (useful for type checking in later functions)
KeyValueCacheTensor = Float[Tensor, "2 batch seq_len n_heads d_head"]

class KeyValueCache(Tensor):
    '''
    This class holds tensors of key and value vectors, to be used for caching.

    If we define it using cfg and batch then it's initialized as empty, but
    we can also define it from kv_cache_entries.
    '''
    @classmethod
    def new_empty(cls, cfg: Config, batch: int = 1) -> "KeyValueCache":
        '''
        Doing a forward pass on a cache created in this way indicates "we don't
        yet have a cache, but we want this forward pass to return a cache".
        Whereas using cache=None in a forward pass indicates we don't want to
        return a cache.
        '''
        shape = (cfg.n_layers, 2, batch, 0, cfg.n_heads, cfg.d_head)
        return cls(*shape).to(device)

    # Define a handful of properties, so they can be referenced directly rather than
    # indexing (which is more likely to lead to mistakes)

    @property
    def k(self) -> Tensor:
        return self[:, 0]

    @property
    def v(self) -> Tensor:
        return self[:, 1]

    @property
    def batch(self) -> int:
        return self.shape[2]

    @property
    def seq_len(self) -> int:
        return self.shape[3]


# Example implementation:
# cfg = cfg
batch = 6
kv_cache = KeyValueCache.new_empty(cfg, batch)

print(f"Shape of all kv-cache = {tuple(kv_cache.shape)}")
print(f"Shape of just k-cache = {tuple(kv_cache.k.shape)}")
for kv_cache_entry in kv_cache:
    print(f"Shape of cache entry for one layer = {tuple(kv_cache_entry.shape)}")
    break
print(f"Batch size = {kv_cache.batch}")
print(f"Current sequence length = {kv_cache.seq_len}")



# %%
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        mean = residual.mean(dim=2, keepdim=True) # (batch, posn)
        var = residual.var(dim=2, keepdim=True, correction=0)
        out1 = residual - mean
        out2 = out1 / t.sqrt(var + self.cfg.layer_norm_eps) # (batch, postn, d_model)
        out3 = out2 * self.w  + self.b
        return out3




# %%
class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]




# %%
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        indices = t.arange(0, seq_len)
        indices = einops.repeat(indices, "position -> batch position", batch=batch)
        return self.W_pos[indices]



# %%
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        ones = t.ones_like(attn_scores)
        mask_to_remove = t.triu(ones, diagonal=1).bool()
        masked_attn_scores = attn_scores.masked_fill_(mask_to_remove, value=self.IGNORE)
        return masked_attn_scores



tests.test_causal_mask(Attention.apply_causal_mask)


# %%
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        q = einops.einsum(self.W_Q, normalized_resid_pre, "n_heads d_model d_head, batch posn d_model -> batch posn n_heads d_head") + self.b_Q
        k = einops.einsum(self.W_K, normalized_resid_pre, "n_heads d_model d_head, batch posn d_model -> batch posn n_heads d_head") + self.b_K
        v = einops.einsum(self.W_V, normalized_resid_pre, "n_heads d_model d_head, batch posn d_model -> batch posn n_heads d_head") + self.b_V

        attn_scores = einops.einsum(q, k, "batch posn_q n_heads d_head, batch posn_k n_heads d_head -> batch n_heads posn_q posn_k")
        scaled_attn_scores = attn_scores / t.sqrt(t.tensor([cfg.d_head], device=device)) 
        masked_attn_scores = self.apply_causal_mask(scaled_attn_scores) # (batch, n_heads, posn_q, posn_k)
        attn_probs = masked_attn_scores.softmax(dim=3)

        z = einops.einsum(attn_probs, v, "batch n_heads posn_q posn_k, batch posn_k n_heads d_head -> batch n_heads posn_q d_head") 

        o = einops.einsum(self.W_O, z, "n_heads d_head d_model, batch n_heads posn d_head -> batch posn n_heads d_model")
        o_summed = einops.reduce(o, "batch posn n_heads d_model -> batch posn d_model", reduction="sum")
        result = o_summed + self.b_O
        return result


    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        # You should copy your solution from earlier
        ones = t.ones_like(attn_scores)
        mask_to_remove = t.triu(ones, diagonal=1).bool()
        masked_attn_scores = attn_scores.masked_fill_(mask_to_remove, value=self.IGNORE)
        return masked_attn_scores


tests.test_causal_mask(Attention.apply_causal_mask)
# tests.rand_float_test(Attention, [2, 4, 768])
# tests.load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])


# %%
class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        pre = einops.einsum(self.W_in, normalized_resid_mid, "d_model d_mlp, batch posn d_model -> batch posn d_mlp") + self.b_in
        activated = gelu_new(pre)
        post = einops.einsum(self.W_out, activated, "d_mlp d_model, batch posn d_mlp -> batch posn d_model") + self.b_out
        return post



# %%
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_model"]:
        resid = resid_pre
        normed1 = self.ln1(resid_pre)
        resid_post_attn = self.attn(normed1)
        resid = resid + resid_post_attn

        normed2 = self.ln2(resid) 
        resid_post_mlp = self.mlp(normed2)
        resid = resid + resid_post_mlp
        return resid




# %%
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)), requires_grad=False)

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return einops.einsum(self.W_U, normalized_resid_final, "d_model d_vocab, batch position d_model -> batch position d_vocab") + self.b_U




# %%
class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        embed = self.embed(tokens) + self.pos_embed(tokens)

        resid = embed
        for block in self.blocks:
            resid = block(resid)

        normed_post_block = self.ln_final(resid)
        logits = self.unembed(normed_post_block)
        return logits

# %%
class TransformerSampler:
    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs) -> str:
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an end-of-sequence token. kwargs are
        passed to sample_next_token, to give detailed instructions on how new tokens are chosen. 
        Pass `seed` to make generation reproducible.
        """
        self.model.eval()
        seed = kwargs.pop("seed", None)
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)


        current_input_ids = self.tokenizer.encode(prompt, return_tensors="pt") # (1, seq_len)

        for _ in range(max_tokens_generated):

            logits = self.model(current_input_ids) # (1, seq, vocab)
            last_logits = logits[0, -1] # (d_vocab)

            sampled_token = self.sample_next_token(current_input_ids[0], last_logits, **kwargs)

            current_input_ids = t.concat([current_input_ids, t.tensor([[sampled_token]])], dim=-1) # (1, seq_len + 1)

            if sampled_token == self.tokenizer.eos_token_id:
                break


        decoded = self.tokenizer.decode(current_input_ids[0], skip_special_tokens=True)
        return decoded


    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, " seq_len"],
        logits: Float[Tensor, "d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
    ) -> int:
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert logits.ndim == 1, "logits should be a 1D tensor of shape (d_vocab,)"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Apply all the specialized sampling methods. The frequency penalty goes first, so that it
        # still applies when temperature == 0, and so that its strength doesn't depend on temperature.
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        return logits.argmax().item()

    @staticmethod
    def apply_temperature(logits: Float[Tensor, "d_vocab"], temperature: float) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, " seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        raise NotImplementedError()

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        m = t.distributions.categorical.Categorical(logits=logits)
        return m.sample().item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        raise NotImplementedError()

    @staticmethod
    def sample_top_p(logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
        raise NotImplementedError()

    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int | None = None,
    ) -> list[tuple[float, str]]:
        """
        Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
        prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
        we've generated `num_return_sequences` terminating sequences.
        """
        raise NotImplementedError()


@t.inference_mode()
def sample_with_cache(
    self: TransformerSampler,
    prompt: str,
    max_tokens_generated=100,
    kv_cache: KeyValueCache | None = None,
    verbose=False,
    seed: int | None = None,
    **kwargs
) -> str:

    self.model.eval()
    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)[0]
    if seed is not None:
        np.random.seed(seed)
        t.manual_seed(seed)

    for i in tqdm(range(max_tokens_generated)):
        # Get new logits (make sure we don't pass in more tokens than the model's context length)
        logits, kv_cache = self.model(input_ids[None, -self.cfg.n_ctx:], kv_cache)
        # We only take logits for the last token, because this is what we're sampling
        logits = logits[0, -1]
        # Get next token (as a tensor of size (1,) so we can concat it to input_ids)
        next_token = t.tensor([TransformerSampler.sample_next_token(input_ids, logits, **kwargs)], device=device)
        # Create new input ids string, with shape (old_seq_len + 1,)
        input_ids = t.cat([input_ids, next_token], dim=-1)
        # Print out results, if required
        if verbose:
            print(self.tokenizer.decode(input_ids, skip_special_tokens=True), end="\r")
        # If our new token was the end-of-text token, stop
        if next_token == getattr(self.tokenizer, "eos_token_id", None):
            break

    return self.tokenizer.decode(input_ids, skip_special_tokens=True)


TransformerSampler.sample = sample_with_cache


# %%
device = t.device("cuda") # can also try "cpu"

model = DemoTransformer(Config()).to(device)
model.load_state_dict(reference_gpt2.state_dict(), strict=False);

initial_text = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
# input_ids = tokenizer.encode(initial_text, return_tensors="pt").squeeze()
tokenizer = reference_gpt2.tokenizer
sampler = TransformerSampler(model, tokenizer)

# Run the noncached version
t0 = time.time()
text = sampler.sample(
    initial_text,
    temperature=0.7,
    top_p=0.95,
    seed=0,
)
print(f"Time taken (without cache): {time.time() - t0:.2f} seconds")
rprint(f"Model output:\n\n[bold dark_orange]{text}[/]")

# Run the cached version
t0 = time.time()
text_with_cache = sampler.sample(
    initial_text,
    temperature=0.7,
    top_p=0.95,
    seed=0,
    kv_cache=KeyValueCache.new_empty(sampler.cfg)
)
print(f"Time taken (with cache): {time.time() - t0:.2f} seconds")
rprint(f"Model output:\n\n[bold dark_orange]{text_with_cache}[/]")

# # Check they are the same
assert text == text_with_cache, "Your outputs are different, meaning you've probably made a mistake in your cache implementation (or failed to use random seeds)."
print("Tests passed!")
