"""PyTorch models module for whitebox analyses."""

# Common utilities
from .common import (
    set_random_seed,
    get_device,
    clear_gpu_memory,
    print_gpu_memory_summary,
)

# RoPE utilities
from .rope_utils import (
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)

# Model loading
from .model_loader import (
    get_deepseek_r1,
    get_tokenizer,
)

# Hook management
from .hooks import (
    apply_qwen_attn_mask_hooks,
    remove_qwen_attn_mask_hooks,
)

# Ablation utilities
from .ablation import (
    AttentionHeadAblator,
    generate_with_ablation,
    test_ablation,
)

# Analysis functions
from .analysis import (
    extract_attention_and_logits,
    get_token_logits_for_word,
    analyze_text,
)

__all__ = [
    # Common
    "set_random_seed",
    "get_device",
    "clear_gpu_memory",
    "print_gpu_memory_summary",
    # RoPE
    "rotate_half",
    "apply_rotary_pos_emb",
    "repeat_kv",
    # Model loading
    "get_deepseek_r1",
    "get_tokenizer",
    # Hooks
    "apply_qwen_attn_mask_hooks",
    "remove_qwen_attn_mask_hooks",
    # Ablation
    "AttentionHeadAblator",
    "generate_with_ablation",
    "test_ablation",
    # Analysis
    "extract_attention_and_logits",
    "get_token_logits_for_word",
    "analyze_text",
]