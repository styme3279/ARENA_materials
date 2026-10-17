"""Hook management utilities for attention masking in PyTorch models."""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Callable, Any
from types import MethodType

import torch
import torch.nn as nn

from .rope_utils import apply_rotary_pos_emb, rotate_half, repeat_kv


class HookManager:
    """Manages hooks lifecycle automatically."""
    
    def __init__(self):
        self._hooks = []
        self._original_methods = {}
    
    def register(self, module: nn.Module, hook_fn: Callable) -> None:
        """Register a hook and track it."""
        hook = module.register_forward_hook(hook_fn)
        self._hooks.append(hook)
    
    def register_method_replacement(self, module: nn.Module, method_name: str, new_method: Callable) -> None:
        """Replace a method on a module and track the original."""
        module_id = id(module)
        key = (module_id, method_name)
        
        if key not in self._original_methods:
            self._original_methods[key] = getattr(module, method_name)
        
        setattr(module, method_name, new_method)
    
    def clear(self) -> None:
        """Remove all registered hooks and restore original methods."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        
        for (module_id, method_name), original_method in self._original_methods.items():
            import gc
            for obj in gc.get_objects():
                if id(obj) == module_id and isinstance(obj, nn.Module):
                    setattr(obj, method_name, original_method)
                    break
        
        self._original_methods.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.clear()


# Store original methods globally for restoration (backward compatibility)
original_qwen_forward_methods = {}


class QwenAttentionHookManager(HookManager):
    """Specialized hook manager for Qwen attention masking."""
    
    def __init__(self, model, token_range, layer_2_heads_suppress=None):
        super().__init__()
        self.model = model
        self.token_range = token_range
        self.layer_2_heads_suppress = layer_2_heads_suppress
        self.applied = False
    
    def apply(self):
        """Apply attention masking hooks to the model."""
        if self.applied:
            print("Hooks already applied.")
            return
        
        if self.token_range is None:
            print("No token range specified for masking.")
            return
        
        assert isinstance(self.token_range, list), f"Bad token_range should be list of lists: {self.token_range=}"
        if isinstance(self.token_range[0], int):
            self.token_range = [self.token_range]
        
        target_modules = []
        module_prefix = "model.layers"
        attn_suffix = "self_attn"
        rotary_emb_module = None
        
        if hasattr(self.model, "model") and hasattr(self.model.model, "rotary_emb"):
            rotary_emb_module = self.model.model.rotary_emb
            print(f"Found rotary_emb at model.model.rotary_emb")
            if not callable(rotary_emb_module):
                print(f"Warning: Found rotary_emb module, but it is not callable. RoPE might fail.")
        else:
            print("Warning: Could not automatically find the main rotary_emb module.")
        
        for name, module in self.model.named_modules():
            if name.startswith(module_prefix) and name.endswith(attn_suffix):
                try:
                    layer_idx_str = name.split(".")[2]
                    layer_idx = int(layer_idx_str)
                    if self.layer_2_heads_suppress is None or layer_idx in self.layer_2_heads_suppress:
                        if (
                            hasattr(module, "config")
                            and hasattr(module, "q_proj")
                            and hasattr(module, "k_proj")
                            and hasattr(module, "v_proj")
                            and hasattr(module, "o_proj")
                        ):
                            target_modules.append((name, module, layer_idx))
                        else:
                            missing = [
                                p
                                for p in ["config", "q_proj", "k_proj", "v_proj", "o_proj"]
                                if not hasattr(module, p)
                            ]
                            print(f"Warning: Module {name} missing attributes: {missing}. Skipping.")
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse layer index from module name: {name}. Skipping.")
        
        if not target_modules:
            print("Error: No suitable Qwen2-style attention modules found.")
            return
        
        print(f"Found {len(target_modules)} Qwen2 attention modules to patch.")
        
        for name, attn_module, layer_idx in target_modules:
            heads_mask = self.layer_2_heads_suppress[layer_idx] if self.layer_2_heads_suppress is not None else None
            new_forward = self._create_masked_forward(
                attn_module.forward, layer_idx, rotary_emb_module, heads_mask=heads_mask
            )
            self.register_method_replacement(attn_module, 'forward', MethodType(new_forward, attn_module))
        
        self.applied = True
    
    def _create_masked_forward(self, original_forward_func, current_layer_idx, rotary_module_ref, heads_mask=None):
        """Create a masked forward function for attention module."""
        token_range = self.token_range
        
        def masked_forward(
            self,  # self is the Qwen2Attention instance
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

            bsz, q_len, _ = hidden_states.size()
            config = self.config
            device = hidden_states.device

            num_heads = config.num_attention_heads
            head_dim = config.hidden_size // num_heads
            num_key_value_heads = config.num_key_value_heads
            num_key_value_groups = num_heads // num_key_value_heads
            hidden_size = config.hidden_size

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

            # RoPE Embeddings
            if position_ids is None:
                position_ids = torch.arange(0, q_len, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0)
            else:
                position_ids = position_ids.to(device)

            if rotary_module_ref is not None and callable(rotary_module_ref):
                try:
                    cos, sin = rotary_module_ref(value_states.to(device), position_ids=position_ids)
                    cos = cos.to(device)
                    sin = sin.to(device)
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin, position_ids
                    )
                except Exception as e:
                    print(f"Warning: Layer {current_layer_idx} - Error during RoPE application: {e}.")
            else:
                print(f"Warning: Layer {current_layer_idx} - No valid rotary embedding module reference.")

            # KV Cache Handling
            kv_seq_len = q_len
            if past_key_value is not None:
                print(f"Warning Layer {current_layer_idx}: past_key_value provided unexpectedly.")
                kv_seq_len += past_key_value[0].shape[-2]
                key_states = torch.cat([past_key_value[0].to(device), key_states], dim=2)
                value_states = torch.cat([past_key_value[1].to(device), value_states], dim=2)

            # GQA Handling
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

            # Attention Score Calculation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

            for token_range_ in token_range:
                assert isinstance(token_range_, list), f"Bad token_range should be list of lists: {token_range=}"

                effective_end_pos = min(token_range_[1], kv_seq_len)
                effective_start_pos = min(token_range_[0], effective_end_pos)

                # Apply custom mask
                if effective_start_pos < effective_end_pos:
                    mask_value = torch.finfo(attn_weights.dtype).min
                    if heads_mask is None:
                        attn_weights[..., effective_start_pos:effective_end_pos] = mask_value
                    else:
                        attn_weights[:, heads_mask, :, effective_start_pos:effective_end_pos] = mask_value

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                expected_mask_shape = (bsz, 1, q_len, kv_seq_len)
                
                if attention_mask.shape != expected_mask_shape:
                    if attention_mask.ndim == 2 and attention_mask.shape == (bsz, kv_seq_len):
                        attention_mask = attention_mask[:, None, None, :]
                    elif attention_mask.ndim == 4 and attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1:
                        pass
                    elif attention_mask.shape[2] == 1 and q_len > 1:
                        attention_mask = attention_mask.expand(bsz, 1, q_len, kv_seq_len)
                    else:
                        print(f"Warning Layer {current_layer_idx}: Mismatch attn mask shape.")
                        attention_mask = None

                if attention_mask is not None:
                    if attention_mask.dtype == torch.bool:
                        attention_mask_float = torch.where(
                            attention_mask, 0.0, torch.finfo(attn_weights.dtype).min
                        ).to(attn_weights.dtype)
                    else:
                        attention_mask_float = attention_mask.to(attn_weights.dtype)

                    try:
                        attn_weights = attn_weights + attention_mask_float
                    except RuntimeError as e:
                        print(f"Error Layer {current_layer_idx}: Cannot add attention mask. Error: {e}")

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # Apply dropout only during training
            dropout_p = self.attention_dropout if hasattr(self, "attention_dropout") else config.attention_dropout
            if self.training:
                attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=True)

            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, hidden_size)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights

        return masked_forward


def apply_qwen_attn_mask_hooks(model, token_range, layer_2_heads_suppress=None):
    """
    Applies hooks to Qwen2-style attention modules to mask attention computation.
    Backward compatibility wrapper using QwenAttentionHookManager.
    
    Args:
        model: The model to apply hooks to
        token_range: Token range(s) to mask - can be a single range or list of ranges
        layer_2_heads_suppress: Dict mapping layer indices to lists of head indices to suppress
    """
    global original_qwen_forward_methods
    original_qwen_forward_methods = {}

    if token_range is None:
        print("No token range specified for masking.")
        return
        
    assert isinstance(token_range, list), f"Bad token_range should be list of lists: {token_range=}"
    if isinstance(token_range[0], int):
        token_range = [token_range]

    target_modules = []
    module_prefix = "model.layers"
    attn_suffix = "self_attn"
    rotary_emb_module = None
    
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rotary_emb_module = model.model.rotary_emb
        print(f"Found rotary_emb at model.model.rotary_emb")
        if not callable(rotary_emb_module):
            print(f"Warning: Found rotary_emb module, but it is not callable. RoPE might fail.")
    else:
        print("Warning: Could not automatically find the main rotary_emb module.")

    for name, module in model.named_modules():
        if name.startswith(module_prefix) and name.endswith(attn_suffix):
            try:
                layer_idx_str = name.split(".")[2]
                layer_idx = int(layer_idx_str)
                if layer_2_heads_suppress is None or layer_idx in layer_2_heads_suppress:
                    if (
                        hasattr(module, "config")
                        and hasattr(module, "q_proj")
                        and hasattr(module, "k_proj")
                        and hasattr(module, "v_proj")
                        and hasattr(module, "o_proj")
                    ):
                        target_modules.append((name, module, layer_idx))
                    else:
                        missing = [
                            p
                            for p in ["config", "q_proj", "k_proj", "v_proj", "o_proj"]
                            if not hasattr(module, p)
                        ]
                        print(f"Warning: Module {name} missing attributes: {missing}. Skipping.")
            except (IndexError, ValueError):
                print(f"Warning: Could not parse layer index from module name: {name}. Skipping.")

    if not target_modules:
        print("Error: No suitable Qwen2-style attention modules found.")
        return

    print(f"Found {len(target_modules)} Qwen2 attention modules to patch.")

    def create_masked_forward(
        original_forward_func, current_layer_idx, rotary_module_ref, heads_mask=None
    ):
        def masked_forward(
            self,  # self is the Qwen2Attention instance
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

            bsz, q_len, _ = hidden_states.size()
            config = self.config
            device = hidden_states.device

            num_heads = config.num_attention_heads
            head_dim = config.hidden_size // num_heads
            num_key_value_heads = config.num_key_value_heads
            num_key_value_groups = num_heads // num_key_value_heads
            hidden_size = config.hidden_size

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

            # RoPE Embeddings
            if position_ids is None:
                position_ids = torch.arange(0, q_len, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0)
            else:
                position_ids = position_ids.to(device)

            if rotary_module_ref is not None and callable(rotary_module_ref):
                try:
                    cos, sin = rotary_module_ref(value_states.to(device), position_ids=position_ids)
                    cos = cos.to(device)
                    sin = sin.to(device)
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin, position_ids
                    )
                except Exception as e:
                    print(f"Warning: Layer {current_layer_idx} - Error during RoPE application: {e}.")
            else:
                print(f"Warning: Layer {current_layer_idx} - No valid rotary embedding module reference.")

            # KV Cache Handling
            kv_seq_len = q_len
            if past_key_value is not None:
                print(f"Warning Layer {current_layer_idx}: past_key_value provided unexpectedly.")
                kv_seq_len += past_key_value[0].shape[-2]
                key_states = torch.cat([past_key_value[0].to(device), key_states], dim=2)
                value_states = torch.cat([past_key_value[1].to(device), value_states], dim=2)

            # GQA Handling
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

            # Attention Score Calculation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

            for token_range_ in token_range:
                assert isinstance(token_range_, list), f"Bad token_range should be list of lists: {token_range=}"

                effective_end_pos = min(token_range_[1], kv_seq_len)
                effective_start_pos = min(token_range_[0], effective_end_pos)

                # Apply custom mask
                if effective_start_pos < effective_end_pos:
                    mask_value = torch.finfo(attn_weights.dtype).min
                    if heads_mask is None:
                        attn_weights[..., effective_start_pos:effective_end_pos] = mask_value
                    else:
                        attn_weights[:, heads_mask, :, effective_start_pos:effective_end_pos] = mask_value

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                expected_mask_shape = (bsz, 1, q_len, kv_seq_len)
                
                if attention_mask.shape != expected_mask_shape:
                    if attention_mask.ndim == 2 and attention_mask.shape == (bsz, kv_seq_len):
                        attention_mask = attention_mask[:, None, None, :]
                    elif attention_mask.ndim == 4 and attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1:
                        pass
                    elif attention_mask.shape[2] == 1 and q_len > 1:
                        attention_mask = attention_mask.expand(bsz, 1, q_len, kv_seq_len)
                    else:
                        print(f"Warning Layer {current_layer_idx}: Mismatch attn mask shape.")
                        attention_mask = None

                if attention_mask is not None:
                    if attention_mask.dtype == torch.bool:
                        attention_mask_float = torch.where(
                            attention_mask, 0.0, torch.finfo(attn_weights.dtype).min
                        ).to(attn_weights.dtype)
                    else:
                        attention_mask_float = attention_mask.to(attn_weights.dtype)

                    try:
                        attn_weights = attn_weights + attention_mask_float
                    except RuntimeError as e:
                        print(f"Error Layer {current_layer_idx}: Cannot add attention mask. Error: {e}")

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # Apply dropout only during training
            dropout_p = self.attention_dropout if hasattr(self, "attention_dropout") else config.attention_dropout
            if self.training:
                attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=True)

            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, hidden_size)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights

        return masked_forward

    for name, attn_module, layer_idx in target_modules:
        if name in original_qwen_forward_methods:
            print(f"Warning: Module {name} seems already patched. Skipping.")
            continue
        original_qwen_forward_methods[name] = attn_module.forward
        heads_mask = layer_2_heads_suppress[layer_idx] if layer_2_heads_suppress is not None else None
        attn_module.forward = MethodType(
            create_masked_forward(
                attn_module.forward, layer_idx, rotary_emb_module, heads_mask=heads_mask
            ),
            attn_module,
        )


def remove_qwen_attn_mask_hooks(model):
    """Restores the original forward methods for Qwen attention modules."""
    global original_qwen_forward_methods
    if not original_qwen_forward_methods:
        print("No Qwen hooks seem to be applied or stored.")
        return

    restored_count = 0
    module_prefix = "model.layers"
    attn_suffix = "self_attn"
    
    for name, module in model.named_modules():
        if name.startswith(module_prefix) and name.endswith(attn_suffix):
            if name in original_qwen_forward_methods:
                module.forward = original_qwen_forward_methods[name]
                restored_count += 1

    if restored_count != len(original_qwen_forward_methods):
        print(f"Warning: Attempted to restore {len(original_qwen_forward_methods)} methods, but only restored {restored_count}.")
    else:
        print(f"Restored {restored_count} original Qwen forward methods.")

    original_qwen_forward_methods = {}