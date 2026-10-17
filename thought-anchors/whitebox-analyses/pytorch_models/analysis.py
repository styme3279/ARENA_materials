"""Analysis functions for extracting attention weights and logits from models."""

import time
import warnings
from typing import Dict, List, Optional, Any

import torch
import numpy as np

from .common import set_random_seed, print_gpu_memory_summary, clear_gpu_memory
from .model_loader import get_deepseek_r1, get_tokenizer
from .hooks import apply_qwen_attn_mask_hooks, remove_qwen_attn_mask_hooks


def extract_attention_and_logits(
    model,
    tokenizer,
    text: str,
    model_name: str = "unknown",
    verbose: bool = False,
    return_logits: bool = True,
    attn_layers: Optional[List[int]] = None,
    token_range_to_mask: Optional[List[int]] = None,
    mask_layers: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, Any]:
    """
    Run a forward pass to extract attention weights and logits.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to analyze
        model_name: Name of the model (for model-specific handling)
        verbose: Whether to print debug information
        return_logits: Whether to return the output logits
        attn_layers: Optional list of specific layers to collect attention weights from
        token_range_to_mask: Optional token range to mask attention
        mask_layers: Optional layer indices to apply attention mask
        
    Returns:
        Dictionary containing attention weights, logits, and metadata
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    if verbose:
        print(f"Text: '{text}'")
        print(f"Encoded to {input_ids.shape[1]} tokens")

    logits = None
    hooks_applied = False
    attention_weights = {}
    
    try:
        with torch.no_grad():
            # Determine if we need autocast for gpt-oss models with bfloat16
            use_autocast = False
            autocast_dtype = torch.float16

            model_dtype = next(model.parameters()).dtype if hasattr(model, "parameters") else torch.float32
            if "gpt-oss" in model_name and model_dtype == torch.bfloat16:
                use_autocast = True
                autocast_dtype = torch.bfloat16
                if verbose:
                    print(f"Using autocast with bfloat16 for GPT-OSS model")

            if use_autocast:
                autocast_context = torch.amp.autocast("cuda", dtype=autocast_dtype)
            else:
                autocast_context = warnings.catch_warnings()

            with autocast_context:
                if not use_autocast:
                    warnings.filterwarnings("ignore", message=".*does not support `output_attentions=True`.*")
                    
                if token_range_to_mask and mask_layers:
                    apply_qwen_attn_mask_hooks(model, token_range_to_mask, layer_2_heads_suppress=mask_layers)
                    hooks_applied = True

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=attn_layers is None or len(attn_layers) > 0,
                    return_dict=True,
                    use_cache=False,
                    output_hidden_states=False,
                )

            if token_range_to_mask is None:
                if hasattr(outputs, "attentions") and outputs.attentions is not None:
                    for layer_idx, attn_weights in enumerate(outputs.attentions):
                        if attn_layers is not None and layer_idx not in attn_layers:
                            continue
                        attention_weights[layer_idx] = attn_weights.detach().cpu()
                else:
                    if attn_layers is not None:
                        print("Warning: Requested attention layers but 'outputs.attentions' not found or is None.")

            if return_logits and hasattr(outputs, "logits"):
                logits = outputs.logits.detach().cpu()
            else:
                if return_logits:
                    print("Warning: Requested logits but 'outputs.logits' not found.")

    except Exception as e:
        print(f"WARNING: Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if hooks_applied:
            remove_qwen_attn_mask_hooks(model)

    if logits is not None:
        logits = logits.detach().cpu().numpy()
    clear_gpu_memory()

    all_tokens = input_ids[0].tolist()
    token_texts = tokenizer.convert_ids_to_tokens(all_tokens)

    result = {
        "text": text,
        "tokens": all_tokens,
        "token_texts": token_texts,
        "input_length": len(all_tokens),
        "attention_weights": attention_weights,
    }

    if logits is not None:
        result["logits"] = logits

    clear_gpu_memory()
    return result


def get_token_logits_for_word(logits: np.ndarray, word: str, model_name: str = "qwen-14b") -> np.ndarray:
    """
    Get logits for a specific word token.
    
    Args:
        logits: Logits array
        word: Word to get logits for
        model_name: Name of the model
        
    Returns:
        Logits for the word token
    """
    tokenizer = get_tokenizer(model_name)
    
    tokens = tokenizer.encode(word, add_special_tokens=False)
    assert len(tokens) == 1, f"Word {word} has {len(tokens)} tokens"
    word_logits = logits[0, :, tokens[0]]
    return word_logits


def analyze_text(
    text: str,
    model_name: str = "qwen-14b",
    seed: int = 0,
    float32: bool = False,
    return_logits: bool = False,
    attn_layers: Optional[List[int]] = None,
    verbose: bool = False,
    token_range_to_mask: Optional[List[int]] = None,
    layers_to_mask: Optional[Dict[int, List[int]]] = None,
    device_map: str = "auto",
) -> Dict[str, Any]:
    """
    Analyze a text using a model's forward pass.
    
    Args:
        text: Input text to analyze
        model_name: Name of the model to use
        seed: Random seed for reproducibility
        float32: Use float32 precision
        return_logits: Whether to return logits
        attn_layers: Layers to collect attention from
        verbose: Print debug information
        token_range_to_mask: Token range to mask
        layers_to_mask: Layers and heads to mask
        device_map: Device mapping strategy
        
    Returns:
        Dictionary containing analysis results
    """
    set_random_seed(seed)

    if device_map == "cpu":  # This was related to debugging and testing small models.
        print("Running on CPU!")

    model, tokenizer = get_deepseek_r1(
        model_name,
        float32=float32,
        device_map=device_map,
    )

    if verbose:
        print(f"Analyzing text: {text}")
    else:
        print(f"Analyzing text: {text[:100]}...")

    t_st = time.time()

    result = extract_attention_and_logits(
        model,
        tokenizer,
        text,
        model_name=model_name,
        verbose=verbose,
        return_logits=return_logits,
        attn_layers=attn_layers,
        token_range_to_mask=token_range_to_mask,
        mask_layers=layers_to_mask,
    )

    print_gpu_memory_summary("After forward pass")

    t_end = time.time()
    num_tokens = len(result["token_texts"])
    print(f"Time taken: {t_end - t_st:.2f} seconds (for {num_tokens} tokens)")

    if attn_layers and any(layer in result["attention_weights"] for layer in attn_layers):
        layer = next(layer for layer in attn_layers if layer in result["attention_weights"])
        attention_data = result["attention_weights"][layer]
        num_nans = np.isnan(attention_data).sum()
        p_nan = num_nans / attention_data.size
        assert ~np.isnan(attention_data).any(), f"Attention has NaNs{attention_data.shape=} ({p_nan=:.1%})"

    print(f"\t*** Analysis complete! ({float32=}) ***")

    del model
    clear_gpu_memory()

    return result
