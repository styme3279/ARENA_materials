# %%
import contextlib
import gc
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.express as px
import pytest
import torch
from dotenv import load_dotenv
from IPython.display import display
from jaxtyping import Float, Int
from peft import LoraConfig
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part34_activation_oracles"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

# Disable runtime errors from custom hooks
os.environ["TORCHDYNAMO_DISABLE"] = "1"
# Allow expandable memory segments on CUDA to avoid OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import part34_activation_oracles.tests as tests
import part34_activation_oracles.utils as utils

MAIN = __name__ == "__main__"

dtype = torch.bfloat16
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def print_with_wrap(s: str, width: int = 80):
    """Print text with line wrapping, preserving newlines."""
    out = []
    for line in s.splitlines(keepends=False):
        out.append(textwrap.fill(line, width=width) if line.strip() else line)
    print("\n".join(out))

# %%
# Model configuration
MODEL_NAME = "Qwen/Qwen3-8B"
ORACLE_LORA_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Loading model: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=dtype,
)
model.eval()

# Add dummy adapter for consistent PeftModel API
dummy_config = LoraConfig()
model.add_adapter(dummy_config, adapter_name="default")

print("Model loaded successfully!")

# %%
print(f"Loading oracle LoRA: {ORACLE_LORA_PATH}")
model.load_adapter(ORACLE_LORA_PATH, adapter_name="oracle", is_trainable=False)
print("Oracle loaded successfully!")

# %%
config_dict = model.peft_config["oracle"].to_dict()
config_df = pd.DataFrame(list(config_dict.items()), columns=["Parameter", "Value"])
display(config_df.style.hide(axis="index"))

# %%
# Simple first example
target_prompt_dict = [
    {"role": "user", "content": "What is the capital of France?"},
]
target_prompt = tokenizer.apply_chat_template(
    target_prompt_dict,
    tokenize=False,
    add_generation_prompt=True,
)

# target_inputs = tokenizer([target_prompt], return_tensors="pt").to(model.device)
# input_len = len(target_inputs)
# generated_ids = model.generate(
#     **target_inputs,
#     max_new_tokens=300,
# )
# target_model_generation = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True)

oracle_prompt = "What are you thinking about?"

results = utils.run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=target_prompt,
    target_lora_path=None,  # Using base model
    oracle_prompt=oracle_prompt,
    oracle_lora_path="oracle",  # Our loaded oracle adapter
    oracle_input_type="full_seq",  # Query the full sequence
    generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 50},
)

print(f"Target prompt: {target_prompt}")
print(f"Oracle question: {oracle_prompt}")
print(f"Oracle response: {results.full_sequence_responses[0]}")

# %%

tokenized_target_prompt = tokenizer.encode(target_prompt)
str_tokens = [tokenizer.decode([t]) for t in tokenized_target_prompt]
print(list(enumerate(zip(tokenized_target_prompt, str_tokens))))
tokens = tokenizer.encode(target_prompt)

segment_start_idx = tokens.index(tokenizer.encode(" France")[0]) + 1

print(segment_start_idx)
print(f"Running oracle on segment {tokenizer.decode(tokens[segment_start_idx:])!r}")
results = utils.run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=target_prompt,
    target_lora_path=None,  # Using base model
    oracle_prompt=oracle_prompt,
    oracle_lora_path="oracle",  # Our loaded oracle adapter
    oracle_input_type="segment",  # Query the full sequence
    segment_start_idx=segment_start_idx,
    segment_end_idx=None,
    generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 50},
)

print(f"Target prompt: {target_prompt}")
print(f"Oracle question: {oracle_prompt}")
#print(f"Oracle response: {results.token_responses[0]}")
print(f"Oracle response: {results.segment_responses[0]}")
#print(f"Oracle response type {type(results.token_responses[0])}")
# %%
print(tokenizer.encode(target_prompt))
print()
print(tokenizer(target_prompt)['input_ids'])

inputs = tokenizer(target_prompt, return_tensors='pt').to(device)
out = model(**inputs)
logits = out.logits
print(logits.shape)
topk, topk_ids = logits[0,-1].topk(k=10)
print(topk_ids)
#print(topk_ids[0])
print(topk_ids.unsqueeze(-1).shape)
tokenizer.batch_decode(topk_ids.unsqueeze(-1))

# %%
target_prompt_dict = [
    {
        "role": "user",
        "content": "The philosopher who drank hemlock taught a student who founded an academy. That student's most famous pupil was",
    },
]
target_prompt = (
    tokenizer
    .apply_chat_template(
        target_prompt_dict,
        tokenize=False,
        add_generation_prompt=True,)
)

oracle_prompt = "What is the next word and why"

results = utils.run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=target_prompt,
    target_lora_path=None,
    oracle_prompt=oracle_prompt,
    oracle_lora_path="oracle",
    oracle_input_type="tokens",  # Query each token independently
    token_start_idx=0,
    token_end_idx=None,
    generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 100},
)

# Display token-by-token responses
print(f"Target prompt has {results.num_tokens} tokens")
print("\nToken-by-token oracle responses:")
print("=" * 80)

target_tokens = tokenizer.convert_ids_to_tokens(results.target_input_ids)
for i, (token, response) in enumerate(zip(target_tokens, results.token_responses)):
    if response:
        print(f"Token {i:3d} ({token:15s}): {response}")

# %%
ids = tokenizer.encode(target_prompt)
probs = model(torch.tensor([ids]).to(device)).logits[0, 3].softmax(-1)
vals, idxs = probs.topk(10)
for v, i in zip(vals, idxs):
    print(f"{tokenizer.decode(i)!r:20} {v.item():.4f}")


# %%

# We format the target prompt and find where "result = foo(3, 4)" begins
target_prompt_dict = [
    {"role": "user", 
     "content": "def foo(x, y):\n    return x**2 + y**2\n\nresult = foo(3, 4)"},
]
formatted_target_prompt = tokenizer.apply_chat_template(
    target_prompt_dict, tokenize=False, add_generation_prompt=False, enable_thinking=False, continue_final_message=False
)

tokens = tokenizer.encode(formatted_target_prompt)
token_strings = [tokenizer.decode([t]) for t in tokens]
segment_start = next(i for i, tok_str in enumerate(token_strings) if "result" in tok_str.lower())

oracle_prompt = "If the inputs were 2 and 3, what will the result be?"

# YOUR CODE HERE - call utils.run_oracle() with oracle_input_type="segment" and the right segment_start_idx


for i in range(10):
    print(segment_start + i)
    print(token_strings[segment_start+i])

    results = utils.run_oracle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_prompt=formatted_target_prompt,
        target_lora_path=None,
        oracle_prompt=oracle_prompt,
        oracle_lora_path="oracle",
        oracle_input_type="segment",  # Query each token independently
        segment_start_idx=segment_start+i,
        segment_end_idx=None,
        generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 100},

    )

    print(f"Oracle response: {results.segment_responses[0]}")
    response = results.segment_responses[0].lower()
    # assert any(x in response for x in ["7", "seven"]), (
    #     f"Expected '7' or 'seven' in response, got: {results.segment_responses[0]}"
    # )


# %%
# Layer configuration
LAYER_COUNTS = {
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-8B": 36,
    "Qwen/Qwen3-32B": 64,
    "google/gemma-2-9b-it": 42,
    "google/gemma-3-1b-it": 26,
    "meta-llama/Llama-3.2-1B-Instruct": 16,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "meta-llama/Llama-3.3-70B-Instruct": 80,
}

def layer_fraction_to_layer(model_name: str, layer_fraction: float) -> int:
    """Convert a layer fraction (0.0-1.0) to a layer number."""
    max_layers = LAYER_COUNTS[model_name]
    return int(max_layers * layer_fraction)


def get_hf_submodule(model: AutoModelForCausalLM, layer: int) -> torch.nn.Module:
    """
    Gets the residual stream submodule for HuggingFace transformers.

    Args:
        model: The model
        layer: Which layer to hook

    Returns:
        The submodule to hook (the layer's output is the residual stream)
    """
    model_name = model.config._name_or_path
    assert re.search("gemma|mistral|Llama|Qwen", model_name), (
        f"Model name {model_name!r} is not supported. Supported architectures: Gemma, Mistral, Llama, Qwen."
    )
    return model.model.layers[layer]


# Check it works as expected
_ = get_hf_submodule(model, layer=LAYER_COUNTS[MODEL_NAME] - 1)
with pytest.raises(IndexError):
    _ = get_hf_submodule(model, layer=LAYER_COUNTS[MODEL_NAME])

# %%
class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass

# %%
def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, Int[Tensor, "batch seq"]],
    start_offset: int | None,
    end_offset: int | None,
) -> dict[int, Float[Tensor, "batch seq d_model"]]:
    """
    Collect activations from multiple layers using forward hooks.

    Args:
        model: The target model
        submodules: Dict mapping layer number to submodule to hook
        inputs_BL: Tokenized inputs (input_ids, attention_mask)
        start_offset: Start of the token slice (negative index from end). Only used when `end_offset`
            is also non-None; if `end_offset` is None, this must also be None (no slicing is applied).
        end_offset: End of the token slice (negative index from end, exclusive). Set both `start_offset`
            and `end_offset` to non-None values to enable token-position slicing; if both are None,
            the full sequence activations are returned.

    Returns:
        Dict mapping layer → activations tensor [batch, length, d_model]
    """

    if end_offset is not None:
        assert start_offset is not None
        assert start_offset < end_offset
        assert end_offset < 0
        assert start_offset < 0
    else:
        assert start_offset is None

    activations = {}
    handles = []
    submodules_to_layer = {submod: layer for layer, submod in submodules.items()}

    def forward_hook(module, inputs, outputs):
        # print(type(outputs))
        # print(inputs)
        # print(isinstance(outputs, tuple))

        # print(module.layers)
        layer = submodules_to_layer[module]
        if isinstance(outputs, tuple):
            activations[layer], _ = outputs
        else:
            activations[layer] = outputs
        
        if layer > max(submodules.keys()):
            raise EarlyStopException
        if end_offset is not None:
            activations[layer] = activations[layer][:,start_offset:end_offset,:]

    for i, submod in submodules.items():
        handles.append(submod.register_forward_hook(forward_hook))
    
    
    try:
        with torch.no_grad():
            model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected")
        raise
    finally:
        for h in handles:
            h.remove()
    
    return activations

# Test the function
test_prompt = "The capital of France is"
test_inputs = tokenizer(test_prompt, return_tensors="pt", add_special_tokens=False).to(device)

# Extract from layer 18 (50% of 36 layers)
layer = layer_fraction_to_layer(MODEL_NAME, 0.5)
submodules = {layer: get_hf_submodule(model, layer)}

activations = collect_activations_multiple_layers(
    model=model,
    submodules=submodules,
    inputs_BL=test_inputs,
    start_offset=None,
    end_offset=None,
)

print(f"Extracted activations from layer {layer}")
print(f"Shape: {activations[layer].shape}")  # Should be [1, seq_len, d_model]

tests.test_collect_activations_multiple_layers(collect_activations_multiple_layers, model, tokenizer, device)


# %%
SPECIAL_TOKEN = " ?"


def get_introspection_prefix(layer: int, num_positions: int) -> str:
    """Create the prefix for oracle prompts with ? tokens."""
    prefix = f"Layer: {layer}\n"
    prefix += SPECIAL_TOKEN * num_positions
    prefix += " \n"
    return prefix


# Test it
prefix = get_introspection_prefix(layer=18, num_positions=5)
print(f"Introspection prefix:\n{prefix!r}")

# %%
def find_pattern_in_tokens(
    token_ids: list[int],
    special_token_str: str,
    num_positions: int,
    tokenizer: AutoTokenizer,
) -> list[int]:
    """
    Find positions of special token in tokenized sequence.

    Args:
        token_ids: List of token IDs
        special_token_str: The special token string (e.g., " ?")
        num_positions: Expected number of occurrences
        tokenizer: Tokenizer to encode special token

    Returns:
        List of positions where special token appears
    """
    special_token = tokenizer.encode(special_token_str)[0]
    positions = [ind for ind, token_id in enumerate(token_ids) if token_id == special_token]
    return positions


# Test the function
test_text = "Layer: 18\n ? ? ? \nWhat is this?"
test_tokens = tokenizer.encode(test_text, add_special_tokens=False)
positions = find_pattern_in_tokens(test_tokens, SPECIAL_TOKEN, 3, tokenizer)
print(f"Found ? tokens at positions: {positions}")

tests.test_find_pattern_in_tokens(find_pattern_in_tokens, tokenizer)

# %%
@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# def get_hf_activation_steering_hook(
#     vectors: Float[Tensor, "num_pos d_model"],
#     positions: list[int],
#     steering_coefficient: float,
#     device: torch.device,
#     dtype: torch.dtype,
# ) -> Callable:
#     """
#     Create hook that injects activations at specified positions (assumes batch_size=1).

#     Args:
#         vectors: Steering vectors [K, d_model] where K is number of positions
#         positions: List of positions to inject at
#         steering_coefficient: Multiplier for steering strength
#         device: Device for tensors
#         dtype: Data type for steering

#     Returns:
#         Hook function that modifies activations during forward pass
#     """


#     def activation_steering_hook(module, inputs, outputs):

#         # Extract the residual stream tensor from outputs (handle tuple case)
#         #  if isinstance(outputs, tuple):
#             # outputs = outputs[0] 

#         print(f"Output shape {outputs.shape}")

#         assert isinstance(outputs, torch.Tensor)
#         # Verify batch_size is 1 (raise error if not)
#         assert outputs.shape[0] == 1, f"Batch size is not one, shape is {outputs.shape}"
#         # Get the original activations at the specified positions
#         specified_position_acts = outputs[0, positions]
#         # Normalize the steering vectors to have the same norm as the originals
#         specified_position_norms = specified_position_acts.norm()
#         vector_norms = vectors.norm(dim=1)
#         normalized_vectors = (vectors / vector_norms.unsqueeze(-1)) * specified_position_norms.unsqueeze(-1)
#         scaled_vectors = normalized_vectors * steering_coefficient # (num_pos, d_model)
#         # Apply steering coefficient and add to original activations
#         outputs[0, positions] += scaled_vectors
#         # Return modified outputs in the same format (tuple or tensor)
#         return outputs

#     return activation_steering_hook 
@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_hf_activation_steering_hook(
    vectors: Float[Tensor, "num_pos d_model"],
    positions: list[int],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Create hook that injects activations at specified positions (assumes batch_size=1).

    Args:
        vectors: Steering vectors [K, d_model] where K is number of positions
        positions: List of positions to inject at
        steering_coefficient: Multiplier for steering strength
        device: Device for tensors
        dtype: Data type for steering

    Returns:
        Hook function that modifies activations during forward pass
    """
    # Normalize vectors to unit norm
    normed_vectors = torch.nn.functional.normalize(vectors, dim=-1).detach()
    positions_tensor = torch.tensor(positions, dtype=torch.long, device=device)

    def hook_fn(module, _input, output):
        # Extract residual stream tensor
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B, L, d_model = resid_BLD.shape

        if B != 1:
            raise ValueError(f"Expected batch_size=1, got B={B}")

        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        # Check positions are valid
        assert positions_tensor.min() >= 0
        assert positions_tensor.max() < L, f"Position {positions_tensor.max()} >= sequence length {L}"

        # Get original activations at steering positions
        orig_KD = resid_BLD[0, positions_tensor, :]  # [K, d_model]
        norms_K1 = orig_KD.norm(dim=-1, keepdim=True)  # [K, 1]

        # Scale normalized steering vectors by original magnitudes
        steered_KD = (normed_vectors * norms_K1 * steering_coefficient).to(dtype)

        # Inject (add to original)
        resid_BLD[0, positions_tensor, :] = steered_KD.detach() + orig_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


# Test the function
# Create dummy data (batch_size=1)
test_positions = [5, 6, 7]  # Inject at positions 5, 6, 7
test_vectors = torch.randn(len(test_positions), model.config.hidden_size, device=device)

hook_fn = get_hf_activation_steering_hook(
    vectors=test_vectors,
    positions=test_positions,
    steering_coefficient=1.0,
    device=device,
    dtype=dtype,
)

# Create dummy activations
dummy_resid = torch.randn(1, 20, model.config.hidden_size, device=device)
orig_values = dummy_resid[0, test_positions, :].clone()

# Apply hook
modified_resid = hook_fn(None, None, dummy_resid)

# Check modifications occurred
new_values = modified_resid[0, test_positions[0], :]
assert not torch.allclose(orig_values, new_values), "Hook should modify activations"
print("Steering hook test passed!")

tests.test_get_hf_activation_steering_hook(get_hf_activation_steering_hook, device, model.config.hidden_size)
tests.test_get_hf_activation_steering_hook_matches_reference(
    get_hf_activation_steering_hook, device, model.config.hidden_size
)

# Test the function
# Create dummy data (batch_size=1)
test_positions = [5, 6, 7]  # Inject at positions 5, 6, 7
test_vectors = torch.randn(len(test_positions), model.config.hidden_size, device=device)

hook_fn = get_hf_activation_steering_hook(
    vectors=test_vectors,
    positions=test_positions,
    steering_coefficient=1.0,
    device=device,
    dtype=dtype,
)

# Create dummy activations
dummy_resid = torch.randn(1, 20, model.config.hidden_size, device=device)
orig_values = dummy_resid[0, test_positions, :].clone()

# Apply hook
modified_resid = hook_fn(None, None, dummy_resid)

# Check modifications occurred
new_values = modified_resid[0, test_positions[0], :]
assert not torch.allclose(orig_values, new_values), "Hook should modify activations"
print("Steering hook test passed!")

tests.test_get_hf_activation_steering_hook(get_hf_activation_steering_hook, device, model.config.hidden_size)
tests.test_get_hf_activation_steering_hook_matches_reference(
    get_hf_activation_steering_hook, device, model.config.hidden_size
)


# %%
@dataclass
class OracleInput:
    """Simplified datapoint for oracle inference (no training-specific fields)."""

    input_ids: list[int]
    layer: int
    steering_vectors: Float[Tensor, "num_pos d_model"]
    positions: list[int]


@dataclass
class OracleResults:
    oracle_lora_path: str | None
    target_lora_path: str | None
    target_prompt: str
    act_key: str
    oracle_prompt: str
    num_tokens: int
    token_responses: list[str | None]
    full_sequence_responses: list[str]
    segment_responses: list[str]
    target_input_ids: list[int]

# %%
def create_oracle_input(
    prompt: str,
    layer: int,
    num_positions: int,
    tokenizer: AutoTokenizer,
    acts_BD: Float[Tensor, "num_pos d_model"],
) -> OracleInput:
    """
    Create an oracle input for inference.

    Args:
        prompt: Question to ask the oracle
        layer: Layer the activations came from
        num_positions: Number of ? tokens (equals length of acts_BD)
        tokenizer: Tokenizer
        acts_BD: Activation vectors [num_positions, d_model]

    Returns:
        OracleInput ready for generation
    """
    # Add the introspection prefix (with ? tokens) to the prompt
    prefix = get_introspection_prefix(layer, num_positions)
    prompt = prefix + prompt    
    input = [{"role": "user", "content": prompt}]
    
    
    # Format using the chat template with add_generation_prompt=True (ends with <|im_start|>assistant\n)
    input = tokenizer.apply_chat_template(
        input,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        continue_final_message=False,
        return_dict=False
    )
    
    # Find ? token positions in the tokenized sequence
    positions = find_pattern_in_tokens(input, SPECIAL_TOKEN, num_positions, tokenizer)
    
    # Return an OracleInput with all fields filled in
    return OracleInput(input_ids=input, layer=layer, steering_vectors=acts_BD.detach(), positions=positions)


# Test the function
test_activations = torch.randn(3, model.config.hidden_size)
datapoint = create_oracle_input(
    prompt="What is the model thinking about?",
    layer=18,
    num_positions=3,
    tokenizer=tokenizer,
    acts_BD=test_activations,
)

print(f"Created datapoint with {len(datapoint.input_ids)} tokens")
print(f"? tokens at positions: {datapoint.positions}")

tests.test_create_oracle_input(create_oracle_input, tokenizer, model.config.hidden_size)

# %%
def run_oracle(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    target_prompt: str,
    oracle_prompt: str,
    layer_fraction: float = 0.5,
    device: torch.device = device,
) -> str:
    """
    Run oracle query from scratch using components we built.

    Args:
        model: Model with oracle LoRA loaded
        tokenizer: Tokenizer
        target_prompt: Prompt to analyze (already formatted with chat template)
        oracle_prompt: Question to ask about activations
        layer_fraction: Which layer to extract from (as fraction of total, 0.0-1.0)
        device: Device

    Returns:
        Oracle's response as string
    """
    # For oracle sampling
    generation_kwargs = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 50}

    # Tokenize target prompt and extract non-padding positions
    inputs_BL = tokenizer(target_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    model_name = model.config._name_or_path
    act_layer = layer_fraction_to_layer(model_name, layer_fraction)
    submodules = {act_layer: get_hf_submodule(model, act_layer)}

    seq_len = inputs_BL["input_ids"].shape[1]
    attn_mask = inputs_BL["attention_mask"][0]
    real_len = int(attn_mask.sum().item())
    left_pad = seq_len - real_len

    # Step 1: Collect activations from the target model
    model.set_adapter("default")
    acts_by_layer = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        start_offset=None,
        end_offset=None,
    )

    # Extract activations for all non-padding positions
    num_positions = real_len
    acts_BD = acts_by_layer[act_layer][0, left_pad:, :]  # [num_positions, d_model]

    # Step 2: Create oracle input
    datapoint = create_oracle_input(
        prompt=oracle_prompt,
        layer=act_layer,
        num_positions=num_positions,
        tokenizer=tokenizer,
        acts_BD=acts_BD,
    )

    # Step 3: Build steering hook, switch to oracle, generate
    input_ids = torch.tensor([datapoint.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    steering_vectors = datapoint.steering_vectors.to(device)
    positions = datapoint.positions

    injection_layer = 1  # Inject at layer 1 (gives oracle max processing depth)
    injection_submodule = get_hf_submodule(model, injection_layer)

    hook_fn = get_hf_activation_steering_hook(
        vectors=steering_vectors,
        positions=positions,
        steering_coefficient=1.0,
        device=device,
        dtype=dtype,
    )

    model.set_adapter("oracle")

    with add_hook(injection_submodule, hook_fn):
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)

    # Decode response
    generated_tokens = output_ids[:, input_ids.shape[1] :]
    response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return response


# Test our implementation
target_prompt_dict = [{"role": "user", "content": "The capital of France is"}]
target_prompt = tokenizer.apply_chat_template(target_prompt_dict, tokenize=False, add_generation_prompt=True)
oracle_prompt = "What answer will the model give, as a single token?"

our_response = run_oracle(
    model=model,
    tokenizer=tokenizer,
    target_prompt=target_prompt,
    oracle_prompt=oracle_prompt,
    layer_fraction=0.5,
    device=device,
)

print(f"Our implementation response: {our_response!r}")

# Compare to library version
library_results = utils.run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=target_prompt,
    target_lora_path=None,
    oracle_prompt=oracle_prompt,
    oracle_lora_path="oracle",
    oracle_input_type="full_seq",
)
library_response = library_results.full_sequence_responses[0]

print(f"Library response: {library_response!r}")
assert our_response.strip().lower() == library_response.strip().lower()
