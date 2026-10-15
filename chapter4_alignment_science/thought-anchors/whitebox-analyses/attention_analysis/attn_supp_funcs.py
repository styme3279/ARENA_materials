import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set, Union
import time

from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from pkld import pkld

from pytorch_models.model_config import model2layers_heads
from pytorch_models import analyze_text
from .logits_funcs import (
    analyze_text_get_p_logits,
    decompress_logits_for_position,
)
from .receiver_head_funcs import (
    get_problem_text_sentences,
    get_model_rollouts_root,
)
from .tokenizer_funcs import get_raw_tokens
from .attn_funcs import get_sentence_token_boundaries


@pkld
def get_suppression_KL_matrix(
    problem_num: int,
    p_nucleus: float = 0.9999,
    model_name: str = "qwen-14b",
    is_correct: bool = True,
    only_first: Optional[int] = None,
    take_log: bool = True,
) -> Optional[np.ndarray]:
    try:
        text, sentences = get_problem_text_sentences(problem_num, is_correct, model_name)
    except Exception as e:
        print(f"Error loading problem {problem_num}: {e}")
        return None

    layers, heads = model2layers_heads(model_name)
    layers_to_mask = {i: list(range(heads)) for i in range(layers)}

    # Convert sentences to token ranges for compatibility
    sentence_boundaries = get_sentence_token_boundaries(text, sentences, model_name)
    sentence2ranges = {i: boundary for i, boundary in enumerate(sentence_boundaries)}

    text_tokens = get_raw_tokens(text, model_name)
    device_map = "auto"

    # Use CPU for very long sequences on Windows
    if len(text_tokens) > 4200 and os.name == "nt":
        device_map = "cpu"
        print(f"Using CPU for long sequence ({problem_num}): {len(text_tokens)=}")

    kw = {
        "text": text,
        "model_name": model_name,
        "seed": 0,
        "p_nucleus": p_nucleus,
        "float32": model_name == "qwen-15b",
        "token_range_to_mask": None,
        "layers_to_mask": None,
        "device_map": device_map,
    }

    try:
        baseline_data = analyze_text_get_p_logits(**kw)
    except (KeyError, torch.OutOfMemoryError, RuntimeError) as e:
        print(f"CUDA failed: {e}. Sleeping for 10 seconds....")
        time.sleep(10)
        return None

    sentence_sentence_scores = np.full((len(sentence2ranges), len(sentence2ranges)), np.nan)

    for (
        sentence_num,
        token_range,
    ) in tqdm(sentence2ranges.items(), desc=f"Examining sentence2sentence ({problem_num})"):
        kw["token_range_to_mask"] = list(token_range)
        kw["layers_to_mask"] = layers_to_mask
        try:
            s_data = analyze_text_get_p_logits(**kw)
        except (KeyError, torch.OutOfMemoryError, RuntimeError) as e:
            print(f"Probably CUDA failed: {e}. Sleeping for 10 seconds....")
            time.sleep(10)
            return

        KL_log_l = []
        for i in range(len(text_tokens)):
            b_idxs, b_logits = decompress_logits_for_position(baseline_data, i)
            s_idxs, s_logits = decompress_logits_for_position(s_data, i)

            KL_sparse = calculate_kl_divergence_sparse(
                (b_idxs, b_logits), (s_idxs, s_logits), temperature=0.6, epsilon=1e-9
            )
            # if norm_entropy:

            if take_log:
                KL_log = np.log(KL_sparse + 1e-9)
            else:
                KL_log = KL_sparse
            KL_log_l.append(KL_log)
            if np.isnan(KL_log):
                raise ValueError(f"NaN KL log: {KL_log}")

        KL_log_l = np.array(KL_log_l)
        sentence_KL_logs = []
        for sentence_idx_loop, token_range_loop in sentence2ranges.items():
            # Ensure indices are within bounds of KL_log_l
            start_idx = min(token_range_loop[0], len(KL_log_l))
            end_idx = min(token_range_loop[1], len(KL_log_l))
            if only_first is not None:
                if end_idx - start_idx > only_first:
                    end_idx = start_idx + only_first

            if start_idx < end_idx:
                mean_log_kl = np.nanmean(KL_log_l[start_idx:end_idx])  # Use nanmean for safety
            else:
                mean_log_kl = np.nan  # Assign NaN if range is empty or invalid
            sentence_KL_logs.append(mean_log_kl)
            sentence_sentence_scores[
                sentence_idx_loop,
                sentence_num,
            ] = mean_log_kl

    return sentence_sentence_scores


def calculate_kl_divergence_sparse(
    baseline_data: Tuple[np.ndarray, np.ndarray],
    suppressed_data: Tuple[np.ndarray, np.ndarray],
    temperature: float = 0.6,
) -> float:
    """
    Calculates the KL divergence KL(P || Q) between two probability distributions
    derived from sparse top-p logits. Clips small negative results to 0.

    P is derived from baseline_data, Q from suppressed_data.

    Args:
        baseline_data (Tuple[np.ndarray, np.ndarray]): Tuple containing
            (indices, logits) for the baseline distribution (P).
            Indices should be int32, logits float16 or float32.
        suppressed_data (Tuple[np.ndarray, np.ndarray]): Tuple containing
            (indices, logits) for the suppressed distribution (Q).
        temperature (float): Temperature to use for softmax conversion. Defaults to 0.6.
        epsilon (float): Small value (currently unused here, handled internally).

    Returns:
        float: The calculated KL divergence KL(P || Q), guaranteed non-negative.
               Returns np.nan if inputs are invalid or calculation yields NaN/inf.
    """
    b_idxs, b_logits = baseline_data
    s_idxs, s_logits = suppressed_data

    if b_idxs is None or b_logits is None or s_idxs is None or s_logits is None:
        print("Warning: Invalid input data (None found). Cannot calculate KL divergence.")
        return np.nan
    if len(b_idxs) != len(b_logits) or len(s_idxs) != len(s_logits):
        print(
            "Warning: Mismatch between length of indices and logits. Cannot calculate KL divergence."
        )
        return np.nan
    # Allow empty arrays for one side? KL(P||0) is inf if P>0, KL(0||Q) is 0.
    # If both are empty, KL is 0 or NaN? Let's return 0 if both empty, NaN otherwise for now.
    if len(b_idxs) == 0 and len(s_idxs) == 0:
        return 0.0
    if len(b_idxs) == 0 or len(s_idxs) == 0:
        # If P is non-empty and Q is empty -> Div is Inf
        # If P is empty and Q is non-empty -> Div is 0
        # Let's return NaN for simplicity unless explicitly handled otherwise.
        # Returning inf might be more correct if P is non-empty.
        # For now, keep NaN to signal an edge case was hit.
        print("Warning: One distribution has no tokens. Returning NaN.")
        return np.nan

    # Ensure logits are float32 for stable softmax
    b_logits = b_logits.astype(np.float32)
    s_logits = s_logits.astype(np.float32)

    union_indices = np.union1d(b_idxs, s_idxs)
    union_size = len(union_indices)

    idx_to_union_pos = {idx: pos for pos, idx in enumerate(union_indices)}

    min_logit_val = -1e9  # Approx -inf for softmax
    b_logits_union = np.full(union_size, min_logit_val, dtype=np.float32)
    s_logits_union = np.full(union_size, min_logit_val, dtype=np.float32)

    for idx, logit in zip(b_idxs, b_logits):
        b_logits_union[idx_to_union_pos[idx]] = logit
    for idx, logit in zip(s_idxs, s_logits):
        s_logits_union[idx_to_union_pos[idx]] = logit

    b_logits_tensor = torch.from_numpy(b_logits_union)
    s_logits_tensor = torch.from_numpy(s_logits_union)

    log_p = F.log_softmax(b_logits_tensor / temperature, dim=0)
    log_q = F.log_softmax(s_logits_tensor / temperature, dim=0)

    # Check for immediate issues after log_softmax
    if (
        torch.isinf(log_p).any()
        or torch.isnan(log_p).any()
        or torch.isinf(log_q).any()
        or torch.isnan(log_q).any()
    ):
        print("Warning: Inf or NaN detected in log-probabilities. Inputs might be too extreme.")
        return np.nan

    p_dist = torch.exp(log_p)  # Equivalent to softmax(logits/T)

    if torch.isnan(p_dist).any():
        print("Warning: NaN detected in P distribution.")
        return np.nan

    kl_terms = p_dist * (log_p - log_q)

    kl_terms = torch.where(p_dist == 0, torch.tensor(0.0, dtype=kl_terms.dtype), kl_terms)

    if torch.isnan(kl_terms).any():
        print("Warning: NaN detected during KL term calculation.")
        return np.nan

    kl_div = torch.sum(kl_terms)

    if torch.isinf(kl_div):
        print("Warning: KL divergence is infinite.")
        return kl_div.item()  # This will be float('inf')

    kl_div_value = kl_div.item()

    if kl_div_value < 0:
        if kl_div_value < -1e-6:  # Adjust tolerance if needed
            print(
                f"Warning: KL divergence significantly negative ({kl_div_value:.2e}). Clipping to 0.0. This might indicate an issue."
            )
        return 0.0
    else:
        return kl_div_value


if __name__ == "__main__":
    problem_num = 2238
    model_name = "qwen-15b"
    is_correct = True  # Use correct solutions

    output_file = get_suppression_KL_matrix(
        problem_num=problem_num,
        p_nucleus=0.9999,
        model_name=model_name,
        is_correct=is_correct,
    )
