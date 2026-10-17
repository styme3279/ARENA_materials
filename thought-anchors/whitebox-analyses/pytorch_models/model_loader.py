"""Model loading utilities for various transformer models."""

import os
import warnings
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import print_gpu_memory_summary
from .model_config import get_model_path


class ModelLoader:
    """Instance-based model loader with caching support."""
    
    def __init__(self):
        self._model_cache = {}
        self._tokenizer_cache = {}
    
    def get_model(
        self,
        model_name: str = "qwen-14b",
        float32: bool = True,
        device_map: str = "auto",
        do_flash_attn: bool = False,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a model and tokenizer with specified configuration.
        
        Args:
            model_name: Short name of the model to load
            float32: Use float32 precision (if False, uses float16/bfloat16)
            device_map: Device mapping strategy
            do_flash_attn: Use Flash Attention 2
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            ValueError: If model name is unknown
        """
        cache_key = (model_name, float32, device_map, do_flash_attn)
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model, tokenizer = self._load_model(model_name, float32, device_map, do_flash_attn)
        
        self._model_cache[cache_key] = (model, tokenizer)
        
        return model, tokenizer
    
    def _load_model(
        self,
        model_name: str,
        float32: bool,
        device_map: str,
        do_flash_attn: bool,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Internal method to actually load the model."""
        try:
            model_config = get_model_path(model_name)
        except ValueError:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Special case: gpt2_medium
        if model_config.is_special and model_name == "gpt2_medium":
            tokenizer = AutoTokenizer.from_pretrained(model_config.huggingface_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_config.huggingface_path, 
                output_attentions=True
            )
            return model, tokenizer
        
        model_path = model_config.huggingface_path
        model_name_path = None
        
        if model_config.local_path and os.path.exists(model_config.local_path):
            model_name_path = model_config.local_path

        # Suppress specific warnings related to tokenizer and attention mechanisms
        warnings.filterwarnings(
            "ignore", message="Sliding Window Attention is enabled but not implemented"
        )
        warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")

        # Always use the HuggingFace path for tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        actual_model_path = model_name_path if model_name_path is not None else model_path

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "device_map": device_map,
            "sliding_window": None,
            "force_download": False,
        }

        if do_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Enabling Flash Attention!")
        else:
            # Use eager attention to support output_attentions=True
            model_kwargs["attn_implementation"] = "eager"

        if any(x in model_path for x in ["Llama", "DeepSeek-R1", "gpt-oss"]):
            del model_kwargs["sliding_window"]

        if float32:
            model_kwargs["torch_dtype"] = torch.float32
            print("Using float32 as requested")
        elif "gpt-oss" in model_name:
            # GPT-OSS models use bfloat16 by default when not using float32
            model_kwargs["torch_dtype"] = torch.bfloat16
            print("Using bfloat16 for GPT-OSS model (recommended)")
        else:
            model_kwargs["torch_dtype"] = torch.float16
            print("Using float16 for model")

        print_gpu_memory_summary("Before model loading")

        model = AutoModelForCausalLM.from_pretrained(actual_model_path, **model_kwargs)

        print_gpu_memory_summary("After model loading")

        return model, tokenizer
    
    def get_tokenizer(self, model_name: str) -> AutoTokenizer:
        """
        Get tokenizer for a model.
        
        Args:
            model_name: Short name of the model
            
        Returns:
            Tokenizer for the model
            
        Raises:
            ValueError: If model name is unknown
        """
        if model_name in self._tokenizer_cache:
            return self._tokenizer_cache[model_name]
        
        model_config = get_model_path(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_config.huggingface_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self._tokenizer_cache[model_name] = tokenizer
        
        return tokenizer
    
    def clear_cache(self):
        """Clear all cached models and tokenizers."""
        self._model_cache.clear()
        self._tokenizer_cache.clear()


# Create a singleton instance for backward compatibility
_default_loader = ModelLoader()


def get_deepseek_r1(
    model_name: str = "qwen-14b",
    float32: bool = True,
    device_map: str = "auto",
    do_flash_attn: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Backward compatibility wrapper for get_deepseek_r1.
    Uses the default singleton ModelLoader instance.
    """
    return _default_loader.get_model(model_name, float32, device_map, do_flash_attn)


def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Backward compatibility wrapper for get_tokenizer.
    Uses the default singleton ModelLoader instance.
    """
    return _default_loader.get_tokenizer(model_name)