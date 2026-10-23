# %%
ANIMALS = [
    # Pets & farm
    'dog', 'cat', 'horse', 'cow', 'pig', 'sheep', 'goat', 'chicken', 'duck', 'goose',
    'turkey', 'rabbit', 'hamster', 'guinea pig', 'goldfish', 'parrot', 'mouse', 'rat',
    'gerbil', 'ferret', 'chinchilla', 'budgie', 'canary', 'cockatoo', 'donkey', 'mule',
    'pony', 'llama', 'alpaca', 'camel', 'yak',
    # Wild mammals
    'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'bear', 'polar bear', 'panda',
    'red panda', 'wolf', 'fox', 'coyote', 'jackal', 'hyena', 'cheetah', 'leopard',
    'snow leopard', 'jaguar', 'cougar', 'lynx', 'bobcat', 'deer', 'moose', 'elk',
    'reindeer', 'antelope', 'gazelle', 'wildebeest', 'bison', 'buffalo', 'hippopotamus',
    'rhinoceros', 'warthog', 'wild boar', 'tapir', 'okapi', 'monkey', 'gorilla',
    'chimpanzee', 'orangutan', 'baboon', 'lemur', 'koala', 'kangaroo', 'wallaby',
    'wombat', 'platypus', 'Tasmanian devil', 'raccoon', 'skunk', 'squirrel', 'chipmunk',
    'beaver', 'otter', 'badger', 'hedgehog', 'porcupine', 'bat', 'mole', 'opossum',
    'armadillo', 'sloth', 'anteater', 'aardvark', 'pangolin', 'meerkat', 'mongoose',
    'weasel', 'mink', 'wolverine', 'capybara', 'shrew', 'hare', 'mountain goat', 'ibex',
    'prairie dog', 'groundhog',
    # Sea mammals
    'dolphin', 'whale', 'orca', 'narwhal', 'beluga', 'porpoise', 'seal', 'sea lion',
    'walrus', 'manatee', 'sea otter',
    # Birds
    'eagle', 'hawk', 'falcon', 'owl', 'vulture', 'condor', 'osprey', 'crow', 'raven',
    'magpie', 'sparrow', 'robin', 'pigeon', 'dove', 'seagull', 'pelican', 'flamingo',
    'swan', 'stork', 'heron', 'crane', 'penguin', 'ostrich', 'emu', 'kiwi', 'peacock',
    'toucan', 'hummingbird', 'woodpecker', 'kingfisher', 'macaw', 'cardinal',
    'blue jay', 'starling', 'swallow', 'finch', 'puffin', 'albatross', 'cormorant',
    'quail', 'pheasant', 'roadrunner', 'cuckoo', 'nightingale', 'lark', 'kookaburra',
    # Reptiles & amphibians
    'snake', 'python', 'cobra', 'rattlesnake', 'boa constrictor', 'anaconda', 'viper',
    'lizard', 'iguana', 'gecko', 'chameleon', 'Komodo dragon', 'bearded dragon',
    'crocodile', 'alligator', 'turtle', 'sea turtle', 'tortoise', 'frog', 'toad',
    'salamander', 'newt', 'axolotl',
    # Fish & sea life
    'shark', 'whale shark', 'salmon', 'tuna', 'trout', 'cod', 'carp', 'catfish',
    'clownfish', 'swordfish', 'stingray', 'manta ray', 'eel', 'piranha', 'pufferfish',
    'seahorse', 'anglerfish', 'sardine', 'herring', 'mackerel', 'jellyfish', 'octopus',
    'squid', 'starfish', 'crab', 'lobster', 'shrimp', 'oyster', 'clam', 'mussel',
    'sea urchin', 'coral',
    # Insects & other invertebrates
    'bee', 'wasp', 'hornet', 'ant', 'butterfly', 'moth', 'beetle', 'ladybug',
    'dragonfly', 'grasshopper', 'cricket', 'praying mantis', 'stick insect', 'cicada',
    'firefly', 'mosquito', 'fly', 'termite', 'cockroach', 'flea', 'tick', 'spider',
    'tarantula', 'scorpion', 'centipede', 'millipede', 'snail', 'slug', 'earthworm',
    'leech',
]
# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter2_rl"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
try:
    import jaxtyping
except:
    %pip install transformers einops jaxtyping wandb tabulate git+https://github.com/ARENA-education/eindex.git

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
# %%
import gc
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Literal

# Allow expandable memory segments on CUDA to avoid OOMs from fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
import wandb
from jaxtyping import Float, Int
from tabulate import tabulate
from torch import Tensor
from tqdm.auto import tqdm
from eindex import eindex
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Make sure exercises are in the path
chapter = "chapter2_rl"
section = "part4_rlhf"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

from part4_rlhf import tests, tests_lora

# PyTorch defaults to one thread per core. On machines that report many more cores than they let you use (common in
# cloud containers), CPU-side work then spends longer waiting on threads than computing. Adjust if needed.
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
# TF32 matmuls on NVIDIA GPUs: noticeably faster training (no effect on CPU, MPS or pre-Ampere GPUs)
t.set_float32_matmul_precision("high")


MAIN = __name__ == "__main__"
# %%
# LOW_GPU_MEM fires automatically on GPUs with less than 22 GiB (e.g. Colab's 16GB T4), or with no CUDA GPU at all.
LOW_GPU_MEM = not t.cuda.is_available() or t.cuda.get_device_properties(device).total_memory / (1024**3) < 22
BASE_MODEL = "gpt2" if LOW_GPU_MEM else "gpt2-medium"
LORA_MODEL = "Qwen/Qwen3-0.6B-Base" if LOW_GPU_MEM else "Qwen/Qwen3-1.7B-Base"
# bfloat16 halves the memory of the frozen base weights in the LoRA section; GPUs before Ampere (e.g. T4) don't support it
LORA_DTYPE = t.bfloat16 if t.cuda.is_available() and t.cuda.is_bf16_supported() else t.float32
RUN_BASE_RLHF = True
print(f"{LOW_GPU_MEM=}, {BASE_MODEL=}, {LORA_MODEL=}, {LORA_DTYPE=}, {RUN_BASE_RLHF=}")
# %%
@dataclass
class GRPO_Args:
    # Basic / global
    seed: int = 1

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "RLHF"
    wandb_entity: str | None = None

    # Duration of different phases
    total_phases: int = 100
    prompt: str = "This is"
    batch_size: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 2

    # Optimization hyperparameters
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    final_scale: float = 0.1

    # GRPO objective
    clip_coef: float = 0.2
    kl_coef: float = 0.3
    gamma: float = 1.0  # discount for per-token rewards (below 1, rewards are credited mostly to the tokens just before)

    # Base model & sampling arguments
    base_model: str = BASE_MODEL
    gen_len: int = 30
    temperature: float = 1.0

    # Reward function, mapping a list of samples (prompt + completion) to a tensor of rewards
    reward_fn: Callable = lambda samples: t.zeros(len(samples), device=device)
    # Optional per-token reward function, mapping each completion's tokens (decoded one at a time) to a tensor of shape
    # (batch, gen_len). If set, it's used instead of `reward_fn`, and each token only gets credit for later rewards.
    token_reward_fn: Callable | None = None

    def __post_init__(self):
        assert self.total_phases > self.warmup_steps, "total_phases must be greater than warmup_steps"
        assert self.batch_size % self.num_minibatches == 0, "batch_size should be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
# %%
@t.no_grad()
def get_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    batch_size: int,
    gen_len: int,
    temperature: float = 1.0,
) -> tuple[Int[Tensor, "batch seq"], list[str]]:
    """
    Samples `batch_size` completions of `gen_len` tokens, all from the same prompt.

    Args:
        model: the language model to sample from
        tokenizer: the model's tokenizer
        prompt: the prompt to complete
        batch_size: the number of completions to sample
        gen_len: the number of new tokens to generate for each completion
        temperature: the sampling temperature (1.0 means sampling from the model's distribution)

    Returns:
        sample_ids: the token ids of the samples (prompt + completion). The completions are the last `gen_len` tokens.
        samples: the samples as strings (prompt + completion, without special tokens), used for the reward
    """
    prompt_ids = tokenizer(tokenizer.eos_token + prompt, return_tensors="pt").input_ids.to(device).repeat(batch_size, 1)
    sample_ids = model.generate(
        prompt_ids,
        # Without an explicit attention mask, `generate` treats every pad token in the prompt as padding. Our pad token
        # is the end-of-text token that starts the prompt, so it would mask that out and sample from a different
        # context to the one we compute log-probabilities on.
        attention_mask=t.ones_like(prompt_ids),
        do_sample=True,
        temperature=temperature,
        top_k=0,  # no truncation: sample from the model's full distribution
        top_p=1.0,
        max_new_tokens=gen_len,
        eos_token_id=None,  # don't stop at the end-of-text token: always generate exactly gen_len tokens
        pad_token_id=tokenizer.eos_token_id,
    )

    completions = tokenizer.batch_decode(sample_ids[:, -gen_len:], skip_special_tokens=True)
    samples = []
    for completion in completions:
        samples.append(prompt + completion)
    return sample_ids, samples
# %%
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=t.float32).to(device)

sample_ids, samples = get_samples(model, tokenizer, prompt="So long, and thanks for all the", batch_size=5, gen_len=15)

table = [[str(ids.tolist()), repr(sample)] for ids, sample in zip(sample_ids, samples)]
print(tabulate(table, ["Token ids", "Sample"], tablefmt="simple_grid", maxcolwidths=60))
# %%
def reward_fn_char_count(generated_sample: list[str], char: str = ".") -> Float[Tensor, " batch"]:
    """
    Reward function counting the number of instances of a particular character.

    Args:
        generated_sample: the generated samples to evaluate
        char: the character to count

    Returns:
        A float tensor of shape (batch,) giving the count of `char` in each sample.
    """
    out = t.zeros(len(generated_sample), device=device)
    for i, sample in enumerate(generated_sample):
        out[i] = sample.count(char)
    return out


def reward_fn_char_count_per_token(tokens: list[list[str]], char: str = ".") -> Float[Tensor, "batch gen_len"]:
    """
    Per-token version of `reward_fn_char_count`: each generated token is rewarded for the instances of `char` in it.

    Args:0
        tokens: the generated tokens of each sample, decoded one at a time
        char: the character to count

    Returns:
        A float tensor of shape (batch, gen_len) giving the count of `char` in each token.
    """
    return t.tensor([[token.count(char) for token in sample] for sample in tokens], dtype=t.float32, device=device)

import re
# Plurals that the regular English rules in `_plural_forms` would get wrong, keyed by the word that changes (the last
# word of multi-word names, e.g. "mantis" for "praying mantis"). An empty list means the plural is the same word as
# the singular ("three sheep"), which is matched already.
IRREGULAR_PLURALS = {
    "mouse": ["mice"],
    "goose": ["geese"],
    "cow": ["cows", "kine"],
    "wolf": ["wolves"],
    "sheep": [],
    "deer": [],
    "moose": [],
    "buffalo": ["buffaloes", "buffalos"],
    "mosquito": ["mosquitoes", "mosquitos"],
    "flamingo": ["flamingos", "flamingoes"],
    "gecko": ["geckos", "geckoes"],
    "armadillo": ["armadillos"],
    "octopus": ["octopuses", "octopi", "octopodes"],
    "platypus": ["platypuses", "platypi"],
    "hippopotamus": ["hippopotamuses", "hippopotami"],
    "rhinoceros": ["rhinoceroses", "rhinoceri"],
    "mantis": ["mantises", "mantes"],
    "cicada": ["cicadas", "cicadae"],
    "tarantula": ["tarantulas", "tarantulae"],
    "ibex": ["ibexes", "ibices"],
}
 
 
def _plural_forms(name: str) -> list[str]:
    """Plural forms of an animal name, changing only the last word of multi-word names ("guinea pig" -> "guinea pigs")."""
    *head, last = name.split(" ")
    key = last.lower()
    if key in IRREGULAR_PLURALS:
        plurals = IRREGULAR_PLURALS[key]
    elif key.endswith(("s", "x", "z", "ch", "sh")):  # fox -> foxes, finch -> finches, catfish -> catfishes
        plurals = [last + "es"]
    elif key.endswith("y") and key[-2] not in "aeiou":  # pony -> ponies, but turkey -> turkeys
        plurals = [last[:-1] + "ies"]
    else:  # dog -> dogs (for salmon, bison, elk etc. this adds the "-s" variant; the bare plural is matched already)
        plurals = [last + "s"]
    return [" ".join([*head, p]) for p in plurals]
 
 
# Every singular and plural form, longest first so that "whale sharks" is one match rather than "whale" + "sharks".
# Built when this cell runs, so re-run it after editing ANIMALS.
ANIMAL_FORMS = sorted({f for a in ANIMALS for f in [a, *_plural_forms(a)]}, key=lambda f: (-len(f), f.lower()))
ANIMAL_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(f) for f in ANIMAL_FORMS) + r")\b", re.IGNORECASE)
 
 
def reward_fn_animal_count(generated_sample: list[str]) -> Float[Tensor, " batch"]:
    """Counts whole-word, case-insensitive mentions of any animal in `ANIMALS`, singular or plural, in each sample."""
    return t.tensor([len(ANIMAL_PATTERN.findall(s)) for s in generated_sample], dtype=t.float32, device=device)

# Test your reward function
A = "This is a test."
B = "......"
C = "Whatever"

t.testing.assert_close(reward_fn_char_count([A]), t.tensor([1.0], device=device))
t.testing.assert_close(reward_fn_char_count([A, B, C]), t.tensor([1.0, 6.0, 0.0], device=device))
t.testing.assert_close(reward_fn_char_count([A], " "), t.tensor([3.0], device=device))
print("All tests for `reward_fn_char_count` passed!")

t.testing.assert_close(
    reward_fn_char_count_per_token([["Hi", ".", " Yes", "...", " ok"]]), t.tensor([[0.0, 1.0, 0.0, 3.0, 0.0]], device=device)
)
# A "." is a single byte, so it always sits inside exactly one token: the per-token counts add up to the sequence count
tokens = tokenizer.batch_decode([[i] for i in tokenizer(A + B + C).input_ids])
t.testing.assert_close(reward_fn_char_count_per_token([tokens]).sum(), reward_fn_char_count([A + B + C])[0])
print("All tests for `reward_fn_char_count_per_token` passed!")
# %%
def compute_grpo_advantages(rewards: Float[Tensor, " batch"], eps: float = 1e-6) -> Float[Tensor, " batch"]:
    """
    Computes GRPO advantages: each reward, normalized by the mean and std of the rewards in the group (the whole batch,
    since all our rollouts share one prompt).

    Args:
        rewards: the reward for each sequence
        eps: small constant added to the std dev for numerical stability

    Returns:
        The GRPO advantage of each sequence.
    """
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


tests.test_compute_grpo_advantages(compute_grpo_advantages)
# %%
def compute_grpo_token_advantages(
    rewards: Float[Tensor, "batch gen_len"], gamma: float = 1.0, eps: float = 1e-6
) -> Float[Tensor, "batch gen_len"]:
    """
    Computes per-token GRPO advantages from per-token rewards. A token can only affect the rewards from its own position
    onwards, so it's credited with the (discounted) sum of those, the reward-to-go, rather than the whole sequence's
    reward. The baseline is the group's mean reward-to-go at the same position. Everything is divided by a single std
    over the whole batch, since per-position stds would blow up the late positions, where few rewards are left.

    With all of a sequence's reward on its last token and gamma = 1, this matches `compute_grpo_advantages`.

    Args:
        rewards: the reward earned by each generated token
        gamma: discount factor; below 1, each token gets less credit for rewards further in the future
        eps: small constant added to the std dev for numerical stability

    Returns:
        The GRPO advantage of each generated token.
    """
    returns = t.zeros_like(rewards)
    future = t.zeros_like(rewards[:, 0])
    for pos in reversed(range(rewards.shape[1])):
        future = rewards[:, pos] + gamma * future
        returns[:, pos] = future
    centered = returns - returns.mean(dim=0)
    return centered / (centered.std() + eps)


# Sequence-level rewards on the last token reproduce `compute_grpo_advantages` (up to the n-1 in the std)
seq_rewards = t.randn(64)
token_rewards = t.zeros(64, 8)
token_rewards[:, -1] = seq_rewards
t.testing.assert_close(
    compute_grpo_token_advantages(token_rewards),
    compute_grpo_advantages(seq_rewards)[:, None].expand(64, 8),
    rtol=1e-2,
    atol=1e-2,
)
# A reward only gives credit to the tokens up to and including the one that earned it
token_rewards = t.zeros(64, 8)
token_rewards[0, 3] = 1.0
adv = compute_grpo_token_advantages(token_rewards)
assert (adv[0, :4] > 0).all() and (adv[0, 4:] == 0).all()
print("All tests for `compute_grpo_token_advantages` passed!")
# %%
def get_logprobs(
    logits: Float[Tensor, "batch seq vocab"],
    tokens: Int[Tensor, "batch seq"],
    gen_len: int,
) -> Float[Tensor, "batch gen_len"]:
    """
    Returns the log-probabilities of the last `gen_len` tokens (the generated ones), under the model with these logits.

    Args:
        logits: the model's output logits. These only need to cover the last `gen_len + 1` positions of `tokens`, but
            may cover more.
        tokens: the token ids, of which the last `gen_len` are the generated tokens
        gen_len: the number of generated tokens

    Returns:
        The logprobs on the tokens sampled.
    """
    logits_on_gen = logits[:, -gen_len - 1 :]
    tokens_on_gen = tokens[:, -gen_len - 1 :]
    logprobs_on_gen = logits_on_gen.log_softmax(dim=-1)
    return eindex(logprobs_on_gen, tokens_on_gen, "b s [b s+1]")
    


tests.test_get_logprobs(get_logprobs)
# %%
def get_logits(
    model: PreTrainedModel,
    sample_ids: Int[Tensor, "batch seq"],
    gen_len: int,
) -> Float[Tensor, "batch gen_len_plus_1 vocab"]:
    """Runs the model on a batch, returning float32 logits for the last `gen_len + 1` positions only."""
    return model(input_ids=sample_ids, logits_to_keep=gen_len + 1).logits.float()


@t.no_grad()
def get_logprobs_batched(
    model: PreTrainedModel,
    sample_ids: Int[Tensor, "batch seq"],
    gen_len: int,
    chunk_size: int,
) -> Float[Tensor, "batch gen_len"]:
    """Log-probabilities of the generated tokens under `model`, computed in chunks of `chunk_size` sequences."""
    logprobs = []
    for chunk in sample_ids.split(chunk_size):
        logits = get_logits(model, chunk, gen_len)
        logprobs.append(get_logprobs(logits, chunk, gen_len))
    return t.cat(logprobs)
# %%
def calc_kl_penalty_naive(
    logprobs: Float[Tensor, "minibatch_size gen_len d_vocab"],
    ref_logprobs: Float[Tensor, "minibatch_size gen_len d_vocab"],
    kl_coef: float,
) -> Float[Tensor, ""]:
    probs = logprobs.exp()
    kl_div = (probs * (logprobs - ref_logprobs)).sum(-1)  # sum d_vocab

    return kl_coef * kl_div.mean()  # mean batch, seq


tests.test_calc_kl_penalty_naive(calc_kl_penalty_naive)
tests.test_calc_kl_penalty_naive_stability(calc_kl_penalty_naive)
# %%
@t.no_grad()
def kl_diagnostics(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str, k: int = 50):
    """Plot the model's top-k next-token probabilities after `prompt`, to show how concentrated the
    distribution is: almost all the mass sits on a handful of tokens."""
    input_ids = tokenizer(tokenizer.eos_token + prompt, return_tensors="pt").input_ids.to(device)
    logprobs = model(input_ids).logits[0, -1].float().log_softmax(-1)
    top = logprobs.argsort(descending=True)[:k]
    probs = logprobs[top].exp().numpy(force=True)
    labels = [repr(tokenizer.decode([i]))[1:-1] for i in top]

    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=200)
    ax.bar(np.arange(k), probs, color="#378ADD")
    ax.set(ylabel="probability", title=f'top-{k} next-token probs after "{prompt}"')
    ax.set_xticks(np.arange(k), labels, rotation=90, fontsize=9)
    fig.tight_layout()
    return fig
# %%
kl_diagnostics(model, tokenizer, "The cat sat on the")
plt.show()
# %%
def calc_kl_penalty(
    logprobs: Float[Tensor, "minibatch_size gen_len"],
    ref_logprobs: Float[Tensor, "minibatch_size gen_len"],
    kl_coef: float,
) -> Float[Tensor, ""]:
    """
    Computes the KL penalty `kl_coef * D_KL(pi_theta || pi_ref)` from only the generated tokens' logprobs, using the k2
    estimator 0.5 * (log pi_theta - log pi_ref)^2.
    Args:
        logprobs:
            log pi_theta(a_t | s_t) for each generated token (under the new model).
        ref_logprobs:
            log pi_ref(a_t | s_t) for each generated token (under the reference model).
        kl_coef:
            The coefficient of the KL penalty.

    Output:
        The (estimated) KL divergence between the new and reference policies, scaled by kl_coef.
    """
    assert logprobs.shape == ref_logprobs.shape, "logprobs and ref_logprobs must have the same shape"
    return kl_coef * (logprobs - ref_logprobs).pow(2).mean() * 0.5


tests.test_calc_kl_penalty(calc_kl_penalty)
tests.test_calc_kl_penalty_stability(calc_kl_penalty)
# %%
def calc_clipped_surrogate_objective(
    logprobs: Float[Tensor, "minibatch_size gen_len"],
    old_logprobs: Float[Tensor, "minibatch_size gen_len"],
    advantages: Float[Tensor, " minibatch_size"] | Float[Tensor, "minibatch_size gen_len"],
    clip_coef: float,
) -> Float[Tensor, ""]:
    """
    Returns the clipped surrogate objective, suitable for maximisation with gradient ascent.

    Args:
        logprobs:
            log pi_theta of each generated token, under the current policy (with gradients)
        old_logprobs:
            log pi_theta_old of each generated token, under the policy at rollout time
        advantages:
            the advantage of each sequence, which applies to all of its tokens, or of each token
        clip_coef:
            amount of clipping, denoted by epsilon in the GRPO / PPO papers

    Returns:
        The clipped surrogate objective, averaged over all tokens in the minibatch.
    """
    assert logprobs.shape == old_logprobs.shape and advantages.shape in (logprobs.shape[:1], logprobs.shape)

    ratio = (logprobs - old_logprobs).exp()
    if advantages.ndim == 1:
        advantages = advantages[:, None]  # broadcast each sequence's advantage over its tokens

    non_clipped = ratio * advantages
    clipped = ratio.clip(1 - clip_coef, 1 + clip_coef) * advantages

    return t.minimum(non_clipped, clipped).mean()
    


tests.test_calc_clipped_surrogate_objective(calc_clipped_surrogate_objective)
# %%
@dataclass
class ReplayMinibatch:
    """
    Samples from the replay memory.
    """

    sample_ids: Int[Tensor, "minibatch_size seq_len"]
    logprobs: Float[Tensor, "minibatch_size gen_len"]
    ref_logprobs: Float[Tensor, "minibatch_size gen_len"]
    advantages: Float[Tensor, " minibatch_size"] | Float[Tensor, "minibatch_size gen_len"]


class ReplayMemory:
    def __init__(
        self,
        args: GRPO_Args,
        sample_ids: Int[Tensor, "batch_size seq_len"],
        logprobs: Float[Tensor, "batch_size gen_len"],
        ref_logprobs: Float[Tensor, "batch_size gen_len"],
        advantages: Float[Tensor, " batch_size"] | Float[Tensor, "batch_size gen_len"],
    ):
        """Initializes the replay memory, with all the data generated from the rollout phase at once."""
        assert sample_ids.shape[0] == args.batch_size
        assert logprobs.shape == ref_logprobs.shape == (args.batch_size, args.gen_len)
        assert advantages.shape in ((args.batch_size,), (args.batch_size, args.gen_len))

        self.args = args
        self.sample_ids = sample_ids
        self.logprobs = logprobs
        self.ref_logprobs = ref_logprobs
        self.advantages = advantages

    def get_minibatches(self) -> list[ReplayMinibatch]:
        """
        Generates a list of minibatches by randomly sampling from the replay memory. Each sequence
        appears exactly `batches_per_learning_phase` times in total.
        """
        minibatches = []
        for _ in range(self.args.batches_per_learning_phase):
            for indices in t.randperm(self.args.batch_size).reshape(self.args.num_minibatches, -1):
                minibatches.append(
                    ReplayMinibatch(
                        sample_ids=self.sample_ids[indices],
                        logprobs=self.logprobs[indices],
                        ref_logprobs=self.ref_logprobs[indices],
                        advantages=self.advantages[indices],
                    )
                )
        return minibatches
# %%
def get_optimizer_and_scheduler(
    args: GRPO_Args, model: nn.Module
) -> tuple[t.optim.Optimizer, t.optim.lr_scheduler.LambdaLR]:
    """
    Creates an AdamW optimizer over the model's trainable parameters, and an LR scheduler that linearly warms up for
    `warmup_steps` phases, and then linearly decays to `final_scale` over the remaining phases.
    """

    def lr_lambda(step: int) -> float:
        assert step <= args.total_phases, f"Step = {step} should be at most total_phases = {args.total_phases}."
        if step < args.warmup_steps:
            return (step + 1) / (args.warmup_steps + 1)
        else:
            return 1 - (1 - args.final_scale) * (step - args.warmup_steps) / (args.total_phases - args.warmup_steps)

    params = [p for p in model.parameters() if p.requires_grad]
    fused = all(p.is_cuda for p in params)  # the whole update step runs as one GPU kernel
    optimizer = t.optim.AdamW(params, lr=args.lr, maximize=True, fused=fused)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler
# %%
class GRPO_Trainer:
    model: PreTrainedModel
    ref_model: PreTrainedModel | None

    def __init__(self, args: GRPO_Args):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = f"{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"

        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        # dtype=float32 explicitly, since Hugging Face otherwise loads the checkpoint's own dtype (often bfloat16)
        # Both in eval mode, which switches off GPT-2's dropout (see the notes above)
        self.model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=t.float32).to(device).eval()
        self.ref_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=t.float32).to(device).eval()
        self.ref_model.requires_grad_(False)
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(args, self.model)

    def get_ref_logprobs(self, sample_ids: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch gen_len"]:
        """Log-probabilities of the generated tokens under the reference policy."""
        return get_logprobs_batched(self.ref_model, sample_ids, self.args.gen_len, self.args.minibatch_size)

    @t.no_grad()
    def log_rollout(
        self,
        samples: list[str],
        rewards: Float[Tensor, " batch"],
        logprobs: Float[Tensor, "batch gen_len"],
        ref_logprobs: Float[Tensor, "batch gen_len"],
    ) -> None:
        """Prints (and logs to wandb) the mean reward, the KL to the reference policy, and a few samples."""
        # Per-token KL, measured with the k1 estimator averaged over the batch: unbiased, and unlike k3 it doesn't blow up
        # when the policy drifts far from the reference (see the KL section)
        kl = (logprobs - ref_logprobs).mean().item()
        if self.args.use_wandb:
            wandb.log({"mean_reward": rewards.mean().item(), "kl": kl}, step=self.step)

        # The first 3 samples
        table = [[f"{rewards[i].item():.2f}", f"{ref_logprobs[i].sum().item():.2f}", repr(samples[i])] for i in range(3)]
        table = tabulate(table, ["Reward", "Ref logprobs", "Sample"], tablefmt="simple_grid", maxcolwidths=[None, None, 90])
        print(
            f"Phase {self.phase + 1:03}/{self.args.total_phases:03}, Mean reward: {rewards.mean().item():.4f}, "
            f"KL: {kl:.4f}\n{table}\n"
        )

    @t.no_grad()
    def rollout_phase(self) -> ReplayMemory:
        """
        Performs a single rollout phase, returning a ReplayMemory object containing the data generated during this
        phase. All forward passes here should be done without gradients.

        Steps of this function are:
            - Generate samples from our model
            - Get the logprobs of the generated tokens under the current & reference policies
            - Compute the rewards and advantages
            - Log the rollouts, and return the data in a ReplayMemory object
        """
        sample_ids, samples = get_samples(
            self.model,
            self.tokenizer,
            prompt=self.args.prompt,
            batch_size=self.args.batch_size,
            gen_len=self.args.gen_len,
            temperature=self.args.temperature,
        )

        logprobs_m = get_logprobs_batched(self.model, sample_ids, self.args.gen_len, self.args.minibatch_size)
        logprobs_ref = self.get_ref_logprobs(sample_ids)
        if self.args.token_reward_fn is None:
            rewards = self.args.reward_fn(samples)
            advantages = compute_grpo_advantages(rewards)
        else:
            # Decode the generated tokens one at a time, so the reward function can tell which token earned each reward
            tokens = [
                self.tokenizer.batch_decode(ids[:, None], skip_special_tokens=True)
                for ids in sample_ids[:, -self.args.gen_len :]
            ]
            token_rewards = self.args.token_reward_fn(tokens)
            rewards = token_rewards.sum(-1)  # each sequence's total, for logging
            advantages = compute_grpo_token_advantages(token_rewards, self.args.gamma)

        self.log_rollout(samples, rewards, logprobs_m, logprobs_ref)

        return ReplayMemory(
            self.args,
            sample_ids,
            logprobs_m,
            logprobs_ref,
            advantages,
        )

    def compute_grpo_objective(self, minibatch: ReplayMinibatch) -> Float[Tensor, ""]:
        """
        Computes the GRPO objective function to maximize: the clipped surrogate objective minus the KL penalty.

        Steps of this function are:
            - Get the logits for the samples in the minibatch, under the current policy
            - Get the logprobs of the generated tokens
            - Use these to compute both terms of the objective function, and return it
            - Also optionally log stuff to Weights & Biases (provided)
        """
        logits = get_logits(self.model, minibatch.sample_ids, self.args.gen_len)
        logprobs = get_logprobs(logits, minibatch.sample_ids, self.args.gen_len)

        clipped_sur_obj = calc_clipped_surrogate_objective(logprobs, minibatch.logprobs, minibatch.advantages, self.args.clip_coef)
        kl = calc_kl_penalty(logprobs, minibatch.ref_logprobs, self.args.kl_coef)
        obj = clipped_sur_obj - kl

        if self.args.use_wandb:
            with t.no_grad():
                ratio = (logprobs - minibatch.logprobs).exp()
                clipfrac = ((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()
            wandb.log(
                dict(
                    total_steps=self.step,
                    lr=self.scheduler.get_last_lr()[0],
                    clipped_surrogate_objective=clipped_sur_obj.item(),
                    clipfrac=clipfrac,
                    kl_penalty=kl.item(),
                ),
                step=self.step,
            )

        return obj

    def learning_phase(self, memory: ReplayMemory) -> float:
        """
        Performs a learning step on `memory`. This involves the standard training steps
        (i.e. zeroing gradient, computing objective function, doing backprop, stepping optimizer).

        You should also remember the following:
            - Clipping grad norm to the value given in `self.args.max_grad_norm`
            - Incrementing `self.step` by 1 for each minibatch
            - Stepping the scheduler (once per calling of this function)

        Returns the average objective function value over the minibatches as a float for logging.
        """
        total = 0.0
        minibatches = memory.get_minibatches()
        for minibatch in tqdm(minibatches):
            self.optimizer.zero_grad()
            objective = self.compute_grpo_objective(minibatch)
            objective.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total += objective.item()
            self.step += 1
        self.scheduler.step()
        return total / len(minibatches)

    def train(self) -> None:
        """
        Performs a full training run.
        """
        self.step = 0

        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                config=self.args,
            )
        runner = tqdm(range(self.args.total_phases))
        for self.phase in runner:
            memory = self.rollout_phase()
            objective = self.learning_phase(memory)
            runner.set_description(f"Objective: {objective:.4f}")

        if self.args.use_wandb:
            wandb.finish()


tests.test_grpo_trainer(GRPO_Trainer, GRPO_Args)
# %%
_, s = get_samples(model, tokenizer, "This is", 256, 30)
r = reward_fn_animal_count(s)
print(f"{(r > 0).float().mean():.1%} score > 0")
print(*[repr(x) for x, ri in zip(s, r) if ri > 0][:10], sep="\n")
# %%
class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear layer W_0 (with optional bias b), adding a trainable low-rank update:

        h = x W_0 + b + (alpha / rank) * x A B

    with A of shape (in_features, rank) initialized from a Gaussian, and B of shape (rank, out_features) initialized to
    zeros (so the update starts at zero).
    """

    A: Float[nn.Parameter, "d_in rank"]
    B: Float[nn.Parameter, "rank d_out"]

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base, self.rank, self.alpha = base, rank, alpha
        self.scaling = alpha / rank
        self.enabled = True  # if False, the module acts exactly like the frozen base layer
        self.merged = False  # if True, the update has been added into base.weight (see `merge`)
        self.base.requires_grad_(False)
        device = base.weight.device
        # float32 adapters, even when the base layer is in bfloat16
        self.A = nn.Parameter(t.randn(base.in_features, rank, device=device) / base.in_features**0.5)
        self.B = nn.Parameter(t.zeros(rank, base.out_features, device=device))

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        out = self.base(x)
        if (not self.enabled) or self.merged:
            return out
        x = x.to(self.A.dtype)
        x = einops.einsum(x, self.A, "... d_in, d_in rank -> ... rank")
        update = einops.einsum(x, self.B, "... rank, rank d_out -> ... d_out")
        return out + (self.scaling * update).to(out.dtype)

    @t.no_grad()
    def merge(self) -> None:
        """Adds the low-rank update into the base layer's weights, so the forward pass no longer needs A and B."""
        assert not self.merged, "LoRALinear: Attempting to merge an already merged model."
        # nn.Linear stores its weight as (d_out, d_in), the transpose of our W_0, so we output the update transposed too
        delta = self.scaling * einops.einsum(self.A, self.B, "d_in rank, rank d_out -> d_out d_in")
        self.base.weight += delta
        self.merged = True

    @t.no_grad()
    def unmerge(self) -> None:
        """
        Undoes `merge`, subtracting the low-rank update back out of the base layer's weights. If the base weights are in
        a lower precision (like bfloat16), merging and unmerging both round, so this only gets back to W_0 approximately.
        """
        assert self.merged, "LoRALinear: merge must be run before running unmerge."
        delta = self.scaling * einops.einsum(self.A, self.B, "d_in rank, rank d_out -> d_out d_in")
        self.base.weight -= delta
        self.merged = False

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, enabled={self.enabled}, merged={self.merged}"


tests_lora.test_lora_linear(LoRALinear)
tests_lora.test_lora_merge(LoRALinear)
# %%
# Every linear layer in both attention and MLP layers
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def apply_lora(
    model: nn.Module,
    target_modules: tuple[str, ...] = LORA_TARGET_MODULES,
    rank: int = 8,
    alpha: float = 8,
) -> list[LoRALinear]:
    """
    Wraps every nn.Linear whose name (the last part of its path) is one of `target_modules` in a LoRALinear.

    Returns:
        The list of LoRALinear modules added to the model.
    """
    lora_modules = []
    # list(...) takes a snapshot, so we don't loop over the LoRALinear modules we're adding
    for name, module in list(model.named_modules()):
        # `name` is the module's full path, e.g. "model.layers.3.self_attn.q_proj"
        if isinstance(module, nn.Linear) and name.split(".")[-1] in target_modules:
            lora = LoRALinear(module, rank=rank, alpha=alpha)
            model.set_submodule(name, lora)
            lora_modules.append(lora)

    if not lora_modules:
        raise ValueError(
            f"No nn.Linear layers named any of {target_modules} were found. Different models name their layers "
            "differently: print the model to see the names."
        )
    return lora_modules


tests_lora.test_apply_lora(apply_lora)
# %%
@contextmanager
def lora_disabled(model: nn.Module):
    """Context manager that switches off all LoRALinear modules in `model`, so it acts as the original model."""
    lora_modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    # A merged adapter is baked into the base weights, so switching it off wouldn't give us the original model back
    assert not any(m.merged for m in lora_modules), "Can't disable merged adapters: unmerge them first"
    for m in lora_modules:
        m.enabled = False
    try:
        yield model
    finally:
        for m in lora_modules:
            m.enabled = True


@dataclass
class GRPO_LoRA_Args(GRPO_Args):
    base_model: str = LORA_MODEL
    dtype: t.dtype = LORA_DTYPE  # dtype of the frozen base model (the adapters are always float32)
    lora_rank: int = 8
    lora_alpha: float = 8  # the paper sets alpha to the first rank it tries, and keeps it fixed
    target_modules: tuple[str, ...] = LORA_TARGET_MODULES
    lr: float = 2e-4  # LoRA needs a higher learning rate than full fine-tuning


class GRPO_LoRA_Trainer(GRPO_Trainer):
    args: GRPO_LoRA_Args

    def __init__(self, args: GRPO_LoRA_Args):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = f"{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        self.model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=args.dtype).to(device).eval()
        self.model.requires_grad_(False)  # freeze the base model *before* adding the (trainable) adapters
        self.lora_modules = apply_lora(
            self.model, target_modules=args.target_modules, rank=args.lora_rank, alpha=args.lora_alpha
        )
        self.ref_model = None  # the reference policy is self.model with the adapters switched off
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(args, self.model)

    def get_ref_logprobs(self, sample_ids: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch gen_len"]:
        with lora_disabled(self.model):
            return get_logprobs_batched(self.model, sample_ids, self.args.gen_len, self.args.minibatch_size)


tests_lora.test_grpo_trainer_lora(GRPO_LoRA_Trainer, GRPO_LoRA_Args, LoRALinear)
# %%
trainer = None
gc.collect()
t.cuda.empty_cache()
# CUDA errors? reduce batch_size or gen_len
args = GRPO_LoRA_Args(             # frozen in LORA_DTYPE, plus the same rank-8 adapters as the Qwen run
    # Gemma 4 E4B base (8B params with its vision/audio towers, ~15 GiB in bf16). LoRA only lands on the language model:
    # the towers' layers are wrapped in Gemma4ClippableLinear, so apply_lora doesn't match them
    base_model="google/gemma-4-E4B",
    # lr: GRPO_LoRA_Args' default of 2e-4 (was 1e-4 for full fine-tuning of gpt2-medium)
    batches_per_learning_phase=2,  # was 8
    num_minibatches=8,             # ARENA's main-run value
    kl_coef=0.25,                  # was 0.9; raise toward 0.6 if samples get incoherent
    batch_size=256, warmup_steps=10, total_phases=100,
    token_reward_fn=reward_fn_char_count_per_token,  # per-token rewards (reward_fn is ignored when this is set)
    use_wandb=True,                # to see clipfrac
)
trainer = GRPO_LoRA_Trainer(args)
trainer.train()
# %%
# # Testing your setup: kl_coef=0.0 (see the dropdown above for explanation)
# if RUN_BASE_RLHF:
#     print(f"LOW_GPU_MEM = {LOW_GPU_MEM}")
#     model = None  # free the demo model from earlier
#     args = GRPO_Args(
#         kl_coef=0.25,
#         lr=2e-3 if LOW_GPU_MEM else 1e-3,  # gpt2 (small) needs a higher lr to collapse as reliably
#         batch_size=256,
#         num_minibatches=4,
#         batches_per_learning_phase=1,
#         total_phases=50,
#         warmup_steps=10,
#         # gen_len=256,
#         # reward_fn=reward_fn_animal_count,
#         # reward_fn=reward_fn_char_count,
#         token_reward_fn=reward_fn_char_count_per_token,
#     )
#     trainer = GRPO_Trainer(args)
#     trainer.train()
# else:
#     print(f"{RUN_BASE_RLHF=}, skipping test run")
# %%
# %%
# Probe: the longest gen_len your GRPO config fits in GPU memory.
# Per candidate gen_len it runs each memory-heavy stage once with your real sizes (generate 1024 rollouts, logprobs
# under policy + ref, 2 training minibatches), records each stage's peak, and bisects to the OOM boundary.
