"""Common imports and utilities for PyTorch model operations."""

import os
import sys
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def print_gpu_memory_summary(prefix=""):
    """Print GPU memory usage summary."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print(f"{prefix} - Running on CPU") 


# Suppress common warnings
warnings.filterwarnings("ignore", message="Sliding Window Attention is enabled but not implemented")
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def get_device() -> torch.device:
    """Get the appropriate device (cuda or cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
