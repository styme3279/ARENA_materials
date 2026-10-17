"""
Attention analysis modules for analyzing reasoning traces in language models.

This package contains modules for:
- Attention weight extraction and analysis
- Receiver head identification and scoring
- Sentence-level KL divergence analysis
- Suppression analysis
- Tokenization utilities
"""

from .attn_funcs import (
    get_avg_attention_matrix,
    get_vertical_scores,
    get_attention_to_step,
)

from .receiver_head_funcs import (
    get_all_heads_vert_scores,
    get_all_receiver_head_scores,
    get_problem_text_sentences,
    get_model_rollouts_root,
    get_top_k_receiver_heads,
)

from .attn_supp_funcs import (
    get_suppression_KL_matrix,
)

from .tokenizer_funcs import (
    get_tokenizer,
    get_raw_tokens,
)

__all__ = [
    # attn_funcs
    "get_avg_attention_matrix",
    "get_vertical_scores",
    "get_attention_to_step",
    # receiver_head_funcs
    "get_all_heads_vert_scores",
    "get_all_receiver_head_scores",
    "get_problem_text_sentences",
    "get_model_rollouts_root",
    "get_top_k_receiver_heads",
    # attn_supp_funcs
    "get_suppression_KL_matrix",
    # tokenizer_funcs
    "get_tokenizer",
    "get_raw_tokens",
]