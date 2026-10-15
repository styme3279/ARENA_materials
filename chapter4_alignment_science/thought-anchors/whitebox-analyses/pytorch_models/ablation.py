"""Attention head ablation utilities for analyzing model behavior."""

import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import numpy as np
from tqdm import tqdm

from .common import set_random_seed, clear_gpu_memory
from .model_loader import get_deepseek_r1


class AttentionHeadAblator:
    """Ablates specified attention heads during model forward passes."""
    
    def __init__(self, ablation_dict: Dict[int, List[int]], verbose: bool = False):
        """
        Initialize attention head ablator.
        
        Args:
            ablation_dict: Dict mapping layer indices to lists of head indices to ablate
                          e.g., {0: [1, 3], 15: [0, 2, 7]}
            verbose: Whether to print debug information
        """
        self.ablation_dict = ablation_dict
        self.verbose = verbose
        self.hooks = []
        self.applied_count = defaultdict(int)
        self.num_heads = None
        self.head_dim = None
        self.logged_layers = set()  # Track which layers we've logged to avoid spam

    def get_model_config(self, model):
        """Extract attention configuration from the model."""
        config = model.config
        self.num_heads = getattr(config, "num_attention_heads", 32)
        hidden_size = getattr(config, "hidden_size", 4096)
        self.head_dim = hidden_size // self.num_heads

        if self.verbose:
            print(f"Model config: {self.num_heads} heads, {self.head_dim} dim per head")

    def ablation_hook(self, layer_idx: int):
        """Create hook function that ablates specified attention heads."""
        
        def hook_fn(module, input_tuple, output):
            heads_to_ablate = self.ablation_dict.get(layer_idx, [])
            if not heads_to_ablate:
                # No ablation needed - return output unchanged
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
                rest_outputs = output[1:]
            else:
                hidden_states = output
                rest_outputs = ()

            batch_size, seq_len, hidden_dim = hidden_states.shape
            expected_dim = self.num_heads * self.head_dim
            if hidden_dim != expected_dim:
                if self.verbose:
                    print(f"Warning: Hidden dim mismatch at layer {layer_idx}: {hidden_dim} vs {expected_dim}")
                return output

            # Clone to avoid in-place modification
            modified_hidden_states = hidden_states.clone()

            # Reshape to separate heads: (batch, seq, num_heads, head_dim)
            reshaped = modified_hidden_states.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )

            # Ablate specified heads by setting them to zero
            valid_heads_ablated = []
            for head_idx in heads_to_ablate:
                if 0 <= head_idx < self.num_heads:
                    reshaped[:, :, head_idx, :] = 0
                    self.applied_count[f"{layer_idx}_{head_idx}"] += seq_len
                    valid_heads_ablated.append(head_idx)
                elif self.verbose:
                    print(f"Warning: Invalid head index {head_idx} for layer {layer_idx} (max: {self.num_heads-1})")

            # Log only once per layer to avoid spam
            if self.verbose and valid_heads_ablated and layer_idx not in self.logged_layers:
                print(f"Ablating heads {valid_heads_ablated} at layer {layer_idx}")
                self.logged_layers.add(layer_idx)

            modified_hidden_states = reshaped.view(batch_size, seq_len, hidden_dim)

            if rest_outputs:
                return (modified_hidden_states,) + rest_outputs
            else:
                return modified_hidden_states

        return hook_fn

    def register_hooks(self, model):
        """Register hooks to ablate attention heads at specified layers."""
        self.remove_hooks()
        self.applied_count = defaultdict(int)
        self.logged_layers = set()
        self.get_model_config(model)

        if not self.ablation_dict:
            if self.verbose:
                print("No ablation specified - skipping hook registration")
            return

        attention_modules = {}

        # Find attention modules ONLY for layers that need ablation
        for name, module in model.named_modules():
            if ".layers." in name and "self_attn" in name and name.endswith("self_attn"):
                parts = name.split(".layers.")
                if len(parts) > 1 and parts[1]:
                    layer_idx_str = parts[1].split(".")[0]
                    try:
                        layer_idx = int(layer_idx_str)
                        # Only register hooks for layers that actually need ablation
                        if layer_idx in self.ablation_dict and self.ablation_dict[layer_idx]:
                            attention_modules[layer_idx] = module
                    except ValueError:
                        pass

        if self.verbose:
            if attention_modules:
                print(f"Registering ablation hooks for layers: {sorted(attention_modules.keys())}")
                print(f"Ablation plan: {self.ablation_dict}")
            else:
                print("No attention modules found for specified layers")

        for layer_idx, attn_module in attention_modules.items():
            hook = attn_module.register_forward_hook(self.ablation_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        if self.verbose and sum(self.applied_count.values()) > 0:
            print(f"Ablation summary:")
            for key, count in sorted(self.applied_count.items()):
                layer_head = key.split("_")
                if len(layer_head) == 2:
                    layer, head = layer_head
                    print(f"  Layer {layer}, Head {head}: Applied to {count} token positions")


def generate_with_ablation(
    model,
    tokenizer,
    prompt: str,
    ablation_dict: Dict[int, List[int]],
    max_new_tokens: int = 100,
    use_sampling: bool = True,
    temperature: float = 0.7,
    verbose: bool = False,
    seed: int = 42,
    num_responses: int = 1,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Generate text with attention head ablation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        ablation_dict: Dict mapping layer indices to head indices to ablate
        max_new_tokens: Maximum number of new tokens to generate
        use_sampling: Whether to use sampling (True) or greedy decoding (False)
        temperature: Temperature for sampling
        verbose: Print debug information
        seed: Random seed for sampling
        num_responses: Number of responses to generate
        
    Returns:
        Dictionary or list of dictionaries containing generation results
    """
    set_random_seed(seed)

    ablator = AttentionHeadAblator(ablation_dict, verbose=verbose)
    ablator.register_hooks(model)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if use_sampling:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.95,
            })
        else:
            gen_kwargs["do_sample"] = False

        results = []
        
        for i in range(num_responses):
            if num_responses > 1 and verbose:
                print(f"Generating response {i+1}/{num_responses}...")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=False,
                )

            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

            all_tokens = outputs.sequences[0].tolist()
            token_texts = tokenizer.convert_ids_to_tokens(all_tokens)

            result = {
                "prompt": prompt,
                "generated_text": generated_text,
                "full_response": full_response,
                "tokens": all_tokens,
                "token_texts": token_texts,
                "ablation_dict": ablation_dict,
                "num_tokens_generated": len(generated_ids),
            }
            
            results.append(result)

    finally:
        # Always remove hooks to clean up
        ablator.remove_hooks()
        clear_gpu_memory()

    return results if num_responses > 1 else results[0]


def test_ablation(
    model,
    tokenizer,
    prompt: str = "The capital of France is",
    layer_indices: Optional[List[int]] = None,
    head_indices: Optional[List[int]] = None,
    max_new_tokens: int = 50,
    verbose: bool = True,
) -> Tuple[str, str]:
    """
    Test ablation by comparing normal and ablated generation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Test prompt
        layer_indices: Layers to ablate (default: [0, 15])
        head_indices: Heads to ablate in each layer (default: [0, 1, 2])
        max_new_tokens: Maximum new tokens to generate
        verbose: Print results
        
    Returns:
        Tuple of (normal_response, ablated_response)
    """
    if layer_indices is None:
        layer_indices = [0, 15]
    if head_indices is None:
        head_indices = [0, 1, 2]

    ablation_dict = {layer: head_indices for layer in layer_indices}

    print("Generating without ablation...")
    normal_result = generate_with_ablation(
        model, tokenizer, prompt,
        ablation_dict={},  # Empty dict = no ablation
        max_new_tokens=max_new_tokens,
        use_sampling=False,
        verbose=False
    )

    print(f"Generating with ablation (layers {layer_indices}, heads {head_indices})...")
    ablated_result = generate_with_ablation(
        model, tokenizer, prompt,
        ablation_dict=ablation_dict,
        max_new_tokens=max_new_tokens,
        use_sampling=False,
        verbose=verbose
    )

    if verbose:
        print("\n" + "="*50)
        print("COMPARISON:")
        print("="*50)
        print(f"Prompt: {prompt}")
        print(f"\nNormal response:\n{normal_result['generated_text']}")
        print(f"\nAblated response:\n{ablated_result['generated_text']}")
        print("="*50)

    return normal_result['generated_text'], ablated_result['generated_text']

if __name__ == "__main__":
    model, tokenizer = get_deepseek_r1(model_name="qwen-15b")
    test_ablation(model, tokenizer)