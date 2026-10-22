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
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

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

    # Base model & sampling arguments
    base_model: str = BASE_MODEL
    gen_len: int = 30
    temperature: float = 1.0

    # Reward function, mapping a list of samples (prompt + completion) to a tensor of rewards
    reward_fn: Callable = lambda samples: t.zeros(len(samples), device=device)

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
    return t.Tensor([s.count(char) for s in generated_sample]).to(device).to(t.float32)


# Test your reward function
A = "This is a test."
B = "......"
C = "Whatever"

t.testing.assert_close(reward_fn_char_count([A]), t.tensor([1.0], device=device))
t.testing.assert_close(reward_fn_char_count([A, B, C]), t.tensor([1.0, 6.0, 0.0], device=device))
t.testing.assert_close(reward_fn_char_count([A], " "), t.tensor([3.0], device=device))
print("All tests for `reward_fn_char_count` passed!")

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
    return (rewards - rewards.mean()) / (rewards.std() + eps)


tests.test_compute_grpo_advantages(compute_grpo_advantages)



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
    sampled_logits = logits[:, -gen_len-1:-1, :]
    sampled_tokens = tokens[:, -gen_len:].unsqueeze(-1)

    log_logits = t.nn.functional.log_softmax(sampled_logits, dim=-1)

    return t.gather(log_logits, -1, sampled_tokens).squeeze(-1)

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


    return (kl_coef * 0.5 * (logprobs - ref_logprobs) ** 2).mean()

tests.test_calc_kl_penalty(calc_kl_penalty)
tests.test_calc_kl_penalty_stability(calc_kl_penalty)

# %%
eps = 0.2
ratio = np.linspace(0.5, 1.5, 201)
fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
for adv, color in [(1.0, "#378ADD"), (-1.0, "#D85A30")]:
    clipped = np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)
    ax.plot(ratio, clipped, color=color, label=f"$\\hat{{A}} = {adv:+.0f}$")
for x in (1 - eps, 1, 1 + eps):
    ax.axvline(x, color="gray", lw=0.8, ls=":" if x != 1 else "--")
ax.set(xlabel=r"ratio $\rho$", ylabel=r"$J^{\mathrm{CLIP}}$", title=f"Clipped surrogate objective ($\\epsilon = {eps}$)")
ax.legend()
fig.tight_layout()
plt.show()

# %%
def calc_clipped_surrogate_objective(
    logprobs: Float[Tensor, "minibatch_size gen_len"],
    old_logprobs: Float[Tensor, "minibatch_size gen_len"],
    advantages: Float[Tensor, " minibatch_size"],
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
            the advantage of each sequence, which applies to all of its tokens
        clip_coef:
            amount of clipping, denoted by epsilon in the GRPO / PPO papers

    Returns:
        The clipped surrogate objective, averaged over all tokens in the minibatch.
    """
    assert logprobs.shape == old_logprobs.shape and advantages.shape == logprobs.shape[:1]

    p = t.exp(logprobs - old_logprobs)

    advantages = einops.repeat(advantages, "minibatchsize -> minibatchsize gen_len", gen_len=p.shape[-1])

    clip_fn = t.clamp(p, 1-clip_coef, 1+clip_coef)

    out = t.minimum(p*advantages, clip_fn*advantages).mean()

    return out


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
    advantages: Float[Tensor, " minibatch_size"]


class ReplayMemory:
    def __init__(
        self,
        args: GRPO_Args,
        sample_ids: Int[Tensor, "batch_size seq_len"],
        logprobs: Float[Tensor, "batch_size gen_len"],
        ref_logprobs: Float[Tensor, "batch_size gen_len"],
        advantages: Float[Tensor, " batch_size"],
    ):
        """Initializes the replay memory, with all the data generated from the rollout phase at once."""
        assert sample_ids.shape[0] == args.batch_size
        assert logprobs.shape == ref_logprobs.shape == (args.batch_size, args.gen_len)
        assert advantages.shape == (args.batch_size,)

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

        with t.inference_mode():
            # logits = get_logits(self.model, sample_ids, self.args.gen_len)
            # logprobs = get_logprobs(logits, sample_ids, self.args.gen_len)
            logprobs = get_logprobs_batched(self.model, sample_ids, self.args.gen_len, chunk_size=8)
            ref_logits = get_logits(self.ref_model, sample_ids, self.args.gen_len)
            ref_logprobs = self.get_ref_logprobs(sample_ids)

        rewards = self.args.reward_fn(samples)

        advantages = compute_grpo_advantages(rewards)


        self.log_rollout(samples, rewards, logprobs, ref_logprobs)

        replay_buffer = ReplayMemory(self.args, sample_ids, logprobs, ref_logprobs, advantages)
        return replay_buffer

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

        clipped_surrogate_objective = calc_clipped_surrogate_objective(logprobs, minibatch.logprobs, minibatch.advantages, self.args.clip_coef)
        kl_penalty = calc_kl_penalty(logprobs, minibatch.ref_logprobs, self.args.kl_coef)

        objective = clipped_surrogate_objective - kl_penalty

        if self.args.use_wandb:
            with t.no_grad():
                ratio = (logprobs - minibatch.logprobs).exp()
                clipfrac = ((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()
            wandb.log(
                dict(
                    total_steps=self.step,
                    lr=self.scheduler.get_last_lr()[0],
                    clipped_surrogate_objective=clipped_surrogate_objective.item(),
                    clipfrac=clipfrac,
                    kl_penalty=kl_penalty.item(),
                ),
                step=self.step,
            )

        return objective

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

        sum_objective = 0
        num_steps = 0
        for minibatch in memory.get_minibatches():
            self.optimizer.zero_grad()
            objective = self.compute_grpo_objective(minibatch)
            objective.backward()
            t.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=self.args.max_grad_norm)
            self.optimizer.step()
            self.step += 1

            sum_objective += objective
            num_steps += 1
        self.scheduler.step()

        return sum_objective / num_steps

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


# tests.test_grpo_trainer(GRPO_Trainer, GRPO_Args)


# # %%
# # Testing your setup: kl_coef=0.0 (see the dropdown above for explanation)
# if RUN_BASE_RLHF:
#     model = None  # free the demo model from earlier
#     args = GRPO_Args(
#         kl_coef=0.0,
#         lr=2e-3 if LOW_GPU_MEM else 1e-3,  # gpt2 (small) needs a higher lr to collapse as reliably
#         batch_size=64,
#         num_minibatches=1,
#         batches_per_learning_phase=1,
#         total_phases=6,
#         warmup_steps=0,
#         reward_fn=reward_fn_char_count,
#     )
#     trainer = GRPO_Trainer(args)
#     trainer.train()
# else:
#     print(f"{RUN_BASE_RLHF=}, skipping test run")

# %%
# if RUN_BASE_RLHF:

#     trainer = None
#     gc.collect()
#     t.cuda.empty_cache()
#     # CUDA errors? reduce batch_size or gen_len
#     args = GRPO_Args(
#         kl_coef=1.25,
#         lr=1e-4,
#         num_minibatches=8,
#         batches_per_learning_phase=2,
#         warmup_steps=5,
#         total_phases=40,
#         reward_fn=reward_fn_char_count,
#         prompt= "Hello, my fellow ",
#     )
#     trainer = GRPO_Trainer(args)
#     trainer.train()
# else:
#     print(f"{RUN_BASE_RLHF=}, skipping main run")


# %%

from transformers import pipeline

cls_model = AutoModelForSequenceClassification.from_pretrained(
    "lvwerra/distilbert-imdb", dtype=t.float16 if t.cuda.is_available() else t.float32
).to(device)
cls_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")


@t.no_grad()
def reward_fn_sentiment_imdb(
    gen_sample: list[str], direction: Literal["pos", "neg"] = "pos"
) -> Float[Tensor, " batch"]:
    """
    Reward function based on sentiment classification probability from the lvwerra/distilbert-imdb
    model.

    Args:
        gen_sample (list[str]): The generated sample to evaluate.
        direction (str): The sentiment of the reward function, either "pos" or "neg".
    """
    assert direction in ["pos", "neg"], "direction should be either 'pos' or 'neg'"

    pipe = pipeline("text-classification", model="tomh/toxigen_hatebert", truncation=True)

    classifications = []
    for sample in gen_sample:
        classification = pipe(sample)
        classification = classification[0]
        if direction == "pos" and classification["label"] == "POSITIVE":
            classifications.append(classification["score"])
        elif direction == "neg" and classification["label"] == "NEGATIVE":
            classifications.append(classification["score"])   
        else:
            classifications.append(1 - classification["score"])  

    return t.Tensor(classifications).to(device)


# Some samples taken from the IMDB dataset used to finetune this model
samples = [
    "Just finished watching this movie for maybe the 7th or 8th time, picked it up one night previously viewed at Blockbuster and absolutely loved it, I've shown it to 4 people so far and they have enjoyed it as well.",
    "This was the most original movie I've seen in years. If you like unique thrillers that are influenced by film noir, then this is just the right cure for all of those Hollywood summer blockbusters clogging the theaters these days.",
    "I can't believe that those praising this movie herein aren't thinking of some other film.",
    "This film seemed way too long even at only 75 minutes.",
    "Really, I can't believe that I spent $5 on this movie. I am a huge zombie fanatic and thought the movie might be really good. It had zombies in it right? Was I wrong!",
]
classes = ["pos", "pos", "neg", "neg", "neg"]

sentiment = reward_fn_sentiment_imdb(samples, direction="pos").tolist()
table = [[repr(sample), cls, f"{sent:.4f}"] for sample, cls, sent in zip(samples, classes, sentiment)]
print(tabulate(table, ["Sample", "Classification", "Sentiment"], tablefmt="simple_grid", maxcolwidths=[80, None, None]))
tests.test_reward_fn_sentiment_imdb(reward_fn_sentiment_imdb)
# %%
MOVIE_PROMPT = "I thought this movie was"

if RUN_BASE_RLHF:
    trainer = None
    gc.collect()
    t.cuda.empty_cache()
    args = GRPO_Args(
        reward_fn=partial(reward_fn_sentiment_imdb, direction="neg"),
        prompt=MOVIE_PROMPT,
        gen_len=40,
        kl_coef=0.1,
        lr=2e-5 if LOW_GPU_MEM else 1e-5,  # gpt2 (small) plateaus below the target at 1e-5
        warmup_steps=0,
        total_phases=12,
    )
    trainer = GRPO_Trainer(args)
    trainer.train()
# %%
