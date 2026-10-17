"""Rotary Position Embedding (RoPE) utilities for transformer models."""

import torch
from typing import Optional


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with rotated dimensions
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    position_ids: Optional[torch.Tensor] = None, 
    unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to the query and key tensors.
    (Copied from transformers/models/qwen2/modeling_qwen2.py for standalone use)
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine values for RoPE
        sin: Sine values for RoPE
        position_ids: Position IDs (optional)
        unsqueeze_dim: Dimension to unsqueeze for broadcasting
        
    Returns:
        Tuple of (embedded query, embedded key)
    """
    # The latest version unsqueezes the cos/sin tensors needed for broadcast comparisons.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    Used for expanding key/value tensors in grouped-query attention.
    (Copied from transformers/models/qwen2/modeling_qwen2.py for standalone use)
    
    Args:
        hidden_states: Key or value tensor with shape (batch, num_key_value_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each head
        
    Returns:
        Expanded tensor with shape (batch, num_key_value_heads * n_rep, seq_len, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)