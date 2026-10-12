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

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])

print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))


logits, cache = reference_gpt2.run_with_cache(tokens)


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

class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        mean = t.mean(residual, axis=-1, keepdim=True)
        var  = t.var(residual, axis=-1, correction=0, keepdim=True)
        result = (residual - mean)/t.sqrt(var + self.cfg.layer_norm_eps)
        result = result * self.w + self.b
        return result


tests.rand_float_test(LayerNorm, [2, 4, 768])
tests.test_layer_norm_epsilon(LayerNorm, cache["resid_post", 11])
tests.load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])


# %%
class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


tests.test_embed(Embed)
tests.rand_int_test(Embed, [2, 4])
tests.load_gpt2_test(Embed, reference_gpt2.embed, tokens)

# %%
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        pos = einops.repeat(t.arange(tokens.shape[-1]), "a -> b a", b=tokens.shape[0])
        return self.W_pos[pos]

tests.test_pos_embed(PosEmbed)
tests.rand_int_test(PosEmbed, [2, 4])
tests.load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)

# # %%
# class Attention(nn.Module):
#     IGNORE: Float[Tensor, ""]

#     def __init__(self, cfg: Config):
#         super().__init__()
#         self.cfg = cfg
#         self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

#     def apply_causal_mask(
#         self,
#         attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
#     ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
#         """
#         Applies a causal mask to attention scores, and returns masked scores.
#         """
#         neg_inf_mat = t.full(attn_scores.shape, self.IGNORE).to(device)
#         neg_inf_mat = neg_inf_mat.triu(diagonal=1)
#         return t.where(neg_inf_mat.to(t.bool), neg_inf_mat, attn_scores)




# tests.test_causal_mask(Attention.apply_causal_mask)

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
        
        # get keys, queries, values
        project = lambda x, y: einops.einsum(x, y, "n e h, b s e -> b s n h")
        K = project(self.W_K, normalized_resid_pre) + self.b_K
        Q = project(self.W_Q, normalized_resid_pre) + self.b_Q
        V = project(self.W_V, normalized_resid_pre) + self.b_V

        # attn scores from dot product of K and Q
        attn_scores = einops.einsum(Q, K, "b s_q n h, b s_k n h -> b n s_q s_k")

        # scale and mask attn scores
        attn_scores = attn_scores/np.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        # softmax along key_pos dim
        attn_scores = t.softmax(attn_scores, dim=-1)

        # get weighted average of value vectors
        z = einops.einsum(attn_scores, V, "b n s_q s_k, b s_k n h -> b s_q n h")

        # map to output space
        res = einops.einsum(self.W_O, z, "n h e, b s_q n h -> b s_q n e")

        # sum over heads and add b_O
        res = einops.reduce(res, "b s_q n e -> b s_q e", "sum") + self.b_O

        return res

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        neg_inf_mat = t.full(attn_scores.shape, self.IGNORE).to(device)
        neg_inf_mat = neg_inf_mat.triu(diagonal=1)
        return t.where(neg_inf_mat.to(t.bool), neg_inf_mat, attn_scores)


tests.test_causal_mask(Attention.apply_causal_mask)
tests.rand_float_test(Attention, [2, 4, 768])
tests.load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])

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
        x = einops.einsum(self.W_in, normalized_resid_mid, "e d_mlp, b s e -> b s d_mlp") + self.b_in
        x = gelu_new(x)
        x = einops.einsum(self.W_out, x, "d_mlp e, b s d_mlp -> b s e") + self.b_out
        return x


tests.rand_float_test(MLP, [2, 4, 768])
tests.load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])


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
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post

tests.rand_float_test(TransformerBlock, [2, 4, 768])
tests.load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

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

        return einops.einsum(self.W_U, normalized_resid_final, "e v, b s e -> b s v") + self.b_U


tests.test_unembed(Unembed)
tests.rand_float_test(Unembed, [2, 4, 768])
tests.load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])

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
        x = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        x = self.unembed(x)
        return x


tests.rand_int_test(DemoTransformer, [2, 4])
tests.load_gpt2_test(DemoTransformer, reference_gpt2, tokens)
# %%

demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

demo_logits = demo_gpt2(tokens)
# %%
def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs for the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens


pred_log_probs = get_log_probs(demo_logits, tokens)
print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

# %%
test_string = """Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as"""
for i in tqdm(range(100)):
    test_tokens = reference_gpt2.to_tokens(test_string).to(device)
    demo_logits = demo_gpt2(test_tokens)
    test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

print(test_string)

# %%
model_cfg = Config(
    debug=False,
    d_model=32,
    n_heads=16,
    d_head=2,
    d_mlp=32 * 4,
    n_layers=4,
    n_ctx=128,
    d_vocab=reference_gpt2.cfg.d_vocab,
)
model = DemoTransformer(model_cfg)

# %%
@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None
    eval_prompt: str = "Once upon a time"


args = TransformerTrainingArgs()

# %%
dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
# Keep the first 200k of the 2.1M stories, as we only need a subset for training.
dataset = dataset.select(range(200_000))
print(dataset)
print(dataset[0]["text"])

# %%
tokenized_dataset = tokenize_and_concatenate(
    dataset,
    reference_gpt2.tokenizer,
    streaming=False,
    max_length=model.cfg.n_ctx,
    column_name="text",
    add_bos_token=True,
    num_proc=8,
)

dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
train_loader = DataLoader(
    dataset_dict["train"], 
    batch_size=args.batch_size, 
    shuffle=True, 
    #num_workers=4, runs faster, but kills the kernel if interrupted
    #pin_memory=False
)
test_loader = DataLoader(
    dataset_dict["test"], 
    batch_size=args.batch_size, 
    shuffle=False, 
    #num_workers=4, 
    #pin_memory=False
)
# %%
class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer):
        super().__init__()
        self.model = model
        self.args = args
        self.sampler = solutions.TransformerSampler(self.model, reference_gpt2.tokenizer)
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0

        self.train_loader = DataLoader(
            dataset_dict["train"],
            batch_size=args.batch_size,
            shuffle=True,
            #num_workers=4, runs faster, but kills the kernel if interrupted
            #pin_memory=False,
        )
        self.test_loader = DataLoader(
            dataset_dict["test"],
            batch_size=args.batch_size,
            shuffle=False,
            #num_workers=4,
            #pin_memory=False,
        )

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """
        # zero out gradients
        self.optimizer.zero_grad()

        data = batch["tokens"].to(device)
        
        # # calculate loss
        predicted_logits = self.model(data)
        loss = -get_log_probs(predicted_logits, data).mean()
        
        # # backpropagate
        loss.backward()

        # # update parameters
        self.optimizer.step()
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """
        Evaluate the model on the test set and return the accuracy.
        """
        self.model.eval()
        #
        # YOUR CODE HERE - fill in the `evaluate` method
        #
        accuracy = 0.0
        total = 0
        for i, batch in enumerate(self.test_loader):
            data = batch["tokens"].to(device)
            predicted_logits = self.model(data)
            # print("shapee")
            # print(predicted_logits.shape)
            predicted_tokens = t.argmax(predicted_logits.exp(), dim=-1)
            # print(predicted_tokens.shape)
            predictions = predicted_tokens[:, :-1]
            correct = data[:, 1:]
            # print(predictions.shape, correct.shape)
            accuracy = accuracy + (predictions == correct).sum()
            total += data.shape[0] * (data.shape[1]-1)
        accuracy = accuracy/total

        self.model.train()
        return accuracy

    def train(self):
        """
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        """
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        print(self.sampler.sample(self.args.eval_prompt, max_tokens_generated=50))
        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")
                if i >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()
            print("Accuracy: ", accuracy)
            print(self.sampler.sample(self.args.eval_prompt, max_tokens_generated=50))

        wandb.finish()


# See the full run here: https://api.wandb.ai/links/dquarel/nrxuwnv7
model = DemoTransformer(model_cfg).to(device)
args = TransformerTrainingArgs()
trainer = TransformerTrainer(args, model)
trainer.train()
# %%
