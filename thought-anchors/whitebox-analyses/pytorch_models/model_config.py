"""Model configuration utilities for whitebox analyses."""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelArchitecture:
    """Model architecture configuration."""
    layers: int
    heads: int
    notes: Optional[str] = None


@dataclass
class ModelPath:
    """Model path configuration for loading."""
    huggingface_path: str
    local_path: Optional[str] = None
    is_special: bool = False  # For models that need special handling like gpt2


# Model architecture configurations
MODEL_ARCHITECTURES: Dict[str, ModelArchitecture] = {
    # Qwen family models
    "qwen-14b": ModelArchitecture(48, 40),
    "qwen-14b-base": ModelArchitecture(48, 40),
    "qwen-15b": ModelArchitecture(28, 12),
    
    # Qwen3 variants
    "qwen3-8b": ModelArchitecture(36, 32),
    "qwen3-30b-a3b": ModelArchitecture(48, 32, "48 layers, 32 attention heads for Q"),
    "qwen3-0p6b": ModelArchitecture(28, 12, "28 layers, 12 attention heads"),
    "qwen3-235b": ModelArchitecture(28, 12),
    "qwen3-235b-a22b": ModelArchitecture(94, 64, "Unconfirmed - needs verification"),
    
    # QWQ models
    "qwq-32b": ModelArchitecture(64, 40),
    
    # Llama models
    "llama8": ModelArchitecture(32, 32),
    "llama8-base": ModelArchitecture(32, 32),
    "llama-v3p1-8b": ModelArchitecture(32, 32, "32 layers, 32 attention heads"),
    "llama-8b": ModelArchitecture(32, 32),
    
    # GPT models
    "gpt-oss-20b": ModelArchitecture(48, 48, "48 layers, 48 attention heads (estimated for 20B model)"),
    
    # DeepSeek models
    "DeepSeek-R1": ModelArchitecture(61, 128),
}


# Model path configurations - maps short names to HuggingFace paths
MODEL_PATHS: Dict[str, ModelPath] = {
    # Qwen family models
    "qwen-14b": ModelPath("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
    "qwen-14b-base": ModelPath("Qwen/Qwen2.5-14B"),
    "qwen-15b": ModelPath("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    "qwen-32b": ModelPath("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
    
    # Qwen3 variants
    "qwen3-8b": ModelPath("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"),
    "qwen3-30b-a3b": ModelPath("Qwen/Qwen3-30B-A3B"),
    "qwen3-0p6b": ModelPath("Qwen/Qwen3-0.6B"),
    "qwen3-235b": ModelPath("qwen/qwen3-235b-a22b"),
    
    # QWQ models
    "qwq-32b": ModelPath("qwen/qwq-32b"),
    
    # Llama models
    "llama8": ModelPath("deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    "llama8-base": ModelPath("meta-llama/Llama-3.1-8B"),
    "llama-v3p1-8b": ModelPath("meta-llama/Meta-Llama-3.1-8B-Instruct"),
    "llama-8b": ModelPath("deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    
    # GPT models
    "gpt-oss-20b": ModelPath("openai/gpt-oss-20b"),
    "gpt2_medium": ModelPath("gpt2-medium", is_special=True),
    
    # Instruction-tuned models
    "it_qwen-14b": ModelPath("Qwen/Qwen2.5-14B-Instruct"),
    
    # DeepSeek models
    "DeepSeek-R1": ModelPath("deepseek-ai/DeepSeek-R1-0528", local_path="/models/DeepSeek-R1"),
}


def model2layers_heads(model_name: str) -> Tuple[int, int]:
    """
    Get the number of layers and attention heads for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Tuple of (num_layers, num_heads)
        
    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name in MODEL_ARCHITECTURES:
        arch = MODEL_ARCHITECTURES[model_name]
        return arch.layers, arch.heads
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model_info(model_name: str) -> ModelArchitecture:
    """
    Get full architecture information for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelArchitecture object with layers, heads, and notes
        
    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name in MODEL_ARCHITECTURES:
        return MODEL_ARCHITECTURES[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def list_supported_models() -> list[str]:
    """
    Get a list of all supported model names.
    
    Returns:
        List of supported model names
    """
    return sorted(MODEL_ARCHITECTURES.keys())


def get_model_path(model_name: str) -> ModelPath:
    """
    Get the model path configuration for loading.
    
    Args:
        model_name: Short name of the model
        
    Returns:
        ModelPath object with HuggingFace path and optional local path
        
    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name in MODEL_PATHS:
        return MODEL_PATHS[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_huggingface_model_name(model_name: str) -> str:
    """
    Get the HuggingFace model path for a given short name.
    
    Args:
        model_name: Short name of the model
        
    Returns:
        Full HuggingFace model path string
        
    Raises:
        ValueError: If the model name is not recognized
    """
    model_path = get_model_path(model_name)
    return model_path.huggingface_path


def get_models_by_family() -> Dict[str, list[str]]:
    """
    Get models grouped by family.
    
    Returns:
        Dictionary mapping model family to list of model names
    """
    families = {
        "qwen": [],
        "qwen3": [],
        "qwq": [],
        "llama": [],
        "gpt": [],
        "deepseek": [],
        "other": []
    }
    
    for model_name in MODEL_ARCHITECTURES:
        if model_name.startswith("qwen3"):
            families["qwen3"].append(model_name)
        elif model_name.startswith("qwen"):
            families["qwen"].append(model_name)
        elif model_name.startswith("qwq"):
            families["qwq"].append(model_name)
        elif "llama" in model_name.lower():
            families["llama"].append(model_name)
        elif "gpt" in model_name.lower():
            families["gpt"].append(model_name)
        elif "deepseek" in model_name.lower():
            families["deepseek"].append(model_name)
        else:
            families["other"].append(model_name)
    
    # Remove empty families
    return {k: v for k, v in families.items() if v}


if __name__ == "__main__":
    # Test the functions
    print("Supported models:")
    for family, models in get_models_by_family().items():
        print(f"\n{family.upper()}:")
        for model in sorted(models):
            layers, heads = model2layers_heads(model)
            info = get_model_info(model)
            if info.notes:
                print(f"  {model}: {layers} layers, {heads} heads ({info.notes})")
            else:
                print(f"  {model}: {layers} layers, {heads} heads")