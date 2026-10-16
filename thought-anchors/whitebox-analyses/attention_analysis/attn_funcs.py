import os
import sys
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from pytorch_models import analyze_text
from pytorch_models.model_config import model2layers_heads
from .tokenizer_funcs import get_raw_tokens
from tqdm import tqdm

from scipy import stats


def get_attention_matrix(
    text: str, model_name: str, layer: int, head: int, device_map: str = "auto"
) -> np.ndarray:
    """
    Get the attention matrix for a specific layer and head for given text.
    Note: This doesn't cache raw matrices as they can be very large.

    Args:
        text: Input text to analyze
        model_name: Name of the model (e.g., "qwen-14b", "llama8-base")
        layer: Layer index
        head: Head index
        device_map: Device mapping for model loading (default "auto", can be "cpu")

    Returns:
        Attention matrix as numpy array
    """
    result = analyze_text(
        text=text,
        model_name=model_name,
        verbose=False,
        return_logits=False,
        attn_layers=None,
        device_map=device_map,
    )

    if len(result["attention_weights"]) == 0:
        raise ValueError("No attention weights returned")

    matrix = result["attention_weights"][layer][0, head].numpy().astype(np.float32)
    return matrix


def generate_text_hash(text: str, sentences: Optional[List[str]] = None) -> str:
    """
    Generate a unique hash based on text content and optional chunk sentences.

    Args:
        text: The input text
        sentences: Optional list of sentences for chunking

    Returns:
        A hexadecimal hash string (first 16 characters of SHA256)
    """
    if sentences:
        content = text + "|||" + "|||".join(sentences)
    else:
        content = text

    hash_obj = hashlib.sha256(content.encode("utf-8"))
    return hash_obj.hexdigest()[:16]


def get_cache_path(
    cache_dir: Union[str, Path],
    text_id: str,
    model_name: str,
    layer: Union[int, List[int]],
    head: int,
    suffix: str = "",
) -> str:
    """
    Generate cache file path for a specific attention matrix.

    Args:
        cache_dir: Base cache directory
        text_id: Unique text identifier
        model_name: Model name
        layer: Layer index
        head: Head index
        suffix: Additional suffix for filename

    Returns:
        Path to cache file
    """
    if isinstance(layer, list):
        layer = "_".join(map(str, layer))
    filename = f"{layer}_{head}{suffix}.npy"

    Path(os.path.join(cache_dir, model_name, text_id, filename)).parent.mkdir(
        parents=True, exist_ok=True
    )
    return os.path.join(cache_dir, model_name, text_id, filename)


def get_sentence_token_boundaries(
    text: str, sentences: List[str], model_name: str
) -> List[Tuple[int, int]]:
    """
    Get exact token boundaries for sentences within the full text.
    This accounts for tokenization effects where tokens may be different
    when sentences are tokenized together vs separately.

    Args:
        text: Full text containing all sentences
        sentences: List of sentence strings
        model_name: Model name for tokenizer

    Returns:
        List of (start, end) token positions for each sentence
    """
    if not sentences:
        return None

    import re

    def normalize_spaces(s: str) -> str:
        """Replace various Unicode spaces with regular space."""
        return re.sub(r"[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]", " ", s)

    char_positions = []
    search_start = 0

    text_normalized = normalize_spaces(text)

    for sentence in sentences:
        sentence_normalized = normalize_spaces(sentence)

        norm_pos = text_normalized.find(sentence_normalized, search_start)
        if norm_pos == -1:
            sentence_stripped = sentence_normalized.strip()
            norm_pos = text_normalized.find(sentence_stripped, search_start)
            if norm_pos == -1:
                raise ValueError(f"Sentence not found in text: {sentence}")
            norm_end = norm_pos + len(sentence_stripped)
        else:
            norm_end = norm_pos + len(sentence_normalized)

        original_pos = 0
        normalized_count = 0
        actual_start = -1
        actual_end = -1

        for i, char in enumerate(text):
            if normalized_count == norm_pos and actual_start == -1:
                actual_start = i
            if normalized_count == norm_end:
                actual_end = i
                break
            if normalize_spaces(char) == " " or char == text_normalized[normalized_count]:
                normalized_count += 1

        if actual_end == -1 and normalized_count == norm_end:
            actual_end = len(text)

        char_positions.append((actual_start, actual_end))
        search_start = norm_end

    token_boundaries = []

    for char_start, char_end in char_positions:
        if char_start > 0:
            tokens_to_start = len(get_raw_tokens(text[:char_start], model_name))
        else:
            tokens_to_start = 0

        tokens_to_end = len(get_raw_tokens(text[:char_end], model_name))

        token_boundaries.append((tokens_to_start, tokens_to_end))

    return token_boundaries


def _compute_averaged_matrix(
    matrix: np.ndarray, sentence_boundaries: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Helper function to compute averaged matrix from raw matrix and boundaries.

    Args:
        matrix: Raw attention matrix
        sentence_boundaries: List of (start, end) tuples for each sentence

    Returns:
        Averaged matrix where each cell (i,j) is the average attention
        from sentence i to sentence j
    """
    if sentence_boundaries is None:
        return matrix

    n = len(sentence_boundaries)
    result = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        row_start, row_end = sentence_boundaries[i]
        row_start = min(row_start, matrix.shape[0] - 1)
        row_end = min(row_end, matrix.shape[0] - 1)

        if row_start >= row_end:
            continue

        for j in range(n):
            col_start, col_end = sentence_boundaries[j]
            col_start = min(col_start, matrix.shape[1] - 1)
            col_end = min(col_end, matrix.shape[1] - 1)

            if col_start >= col_end:
                continue

            region = matrix[row_start:row_end, col_start:col_end]
            if region.size > 0:
                result[i, j] = np.mean(region)

    return result


def compute_all_attention_matrices(
    text: str,
    model_name: str,
    sentences: Optional[List[str]],
    cache_dir: str = "avg_matrices",
    text_id: Optional[str] = None,
    device_map: str = "auto",
    force_recompute: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Compute attention matrices for all layers and heads at once.
    This is more efficient than computing them one by one.

    Args:
        text: Input text to analyze
        model_name: Name of the model
        sentences: Optional list of sentences to chunk the text by
        cache_dir: Directory to cache matrices
        text_id: Unique identifier for the text (auto-generated if not provided)
        device_map: Device mapping for model loading
        force_recompute: Force recomputation even if cache exists
        verbose: Print progress messages

    Returns:
        True if successful, False otherwise
    """
    n_layers, n_heads = model2layers_heads(model_name)

    if cache_dir and not text_id:
        text_id = generate_text_hash(text, sentences)

    if cache_dir and text_id and not force_recompute:
        all_exist = True
        for layer in range(n_layers):
            for head in range(n_heads):
                cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
                if not os.path.exists(cache_path):
                    all_exist = False
                    break
            if not all_exist:
                break

        if all_exist:
            if verbose:
                print(f"All matrices for {text_id} already exist in cache")
            return True

    if verbose:
        print(f"Computing attention matrices for {text_id}...")


    # Check text length and adjust device if needed
    tokens = get_raw_tokens(text, model_name)

    # if os.name == "nt":  # This was related to debugging and testing small models.
    #     if verbose:
    #         print("Running on Windows, using CPU") 
    #     device_map = "cpu"

    if (
        os.name == "nt"
    ):  # This was related to debugging and testing small models.
        if verbose or True:
            print(f"Running on Windows, using CPU ({len(tokens)=})")
        device_map = "cpu"
        import torch
        torch.backends.cudnn.benchmark = False  # For CPU
        torch.set_num_threads(os.cpu_count())  # Max CPU threads
    elif len(tokens) > 3000:
        if verbose or True:
            print(f"Using CPU for long sequence ({len(tokens)=}):")
        device_map = "cpu"
        import torch
        torch.backends.cudnn.benchmark = False  # For CPU
        torch.set_num_threads(os.cpu_count())  # Max CPU threads

    result = analyze_text(
        text,
        model_name=model_name,
        verbose=verbose,
        float32=True,#model_name == "qwen-15b",
        attn_layers=None,
        return_logits=False,
        device_map=device_map,
    )

    if len(result["attention_weights"]) == 0:
        if verbose:
            print("No attention weights returned")
        return False

    sentence_boundaries = None
    if sentences:
        sentence_boundaries = get_sentence_token_boundaries(text, sentences, model_name)

    for layer in tqdm(range(n_layers), desc="Saving avg. matrices"):
        for head in range(n_heads):
            matrix = result["attention_weights"][layer][0, head].numpy().astype(np.float32)

            if sentence_boundaries:
                matrix = _compute_averaged_matrix(matrix, sentence_boundaries)

            # Only the final matrix, not raw
            if cache_dir and text_id:
                cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, matrix)

    return True


def get_avg_attention_matrix(
    text: str,
    model_name: str,
    layer: int,
    head: int,
    sentences: Optional[List[str]],
    device_map: str = "auto",
    cache_dir: Optional[str] = "attn_cache",
    text_id: Optional[str] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Get averaged attention matrix for a specific layer and head.

    Args:
        text: Input text to analyze
        model_name: Name of the model
        layer: Layer index
        head: Head index
        sentences: Optional list of sentences to chunk the text by
        device_map: Device mapping for model loading
        cache_dir: Directory to cache matrices (if None, no caching)
        text_id: Unique identifier for the text (auto-generated if not provided)
        force_recompute: Force recomputation even if cache exists

    Returns:
        Averaged attention matrix as numpy array
    """
    if cache_dir and not text_id:
        text_id = generate_text_hash(text, sentences)

    if cache_dir and text_id and not force_recompute:
        cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
        if os.path.exists(cache_path):
            return np.load(cache_path)

    if cache_dir and text_id:
        # print(f"Computing attention matrices for {text_id}...")
        success = compute_all_attention_matrices(
            text=text,
            model_name=model_name,
            sentences=sentences,
            cache_dir=cache_dir,
            text_id=text_id,
            device_map=device_map,
            force_recompute=force_recompute,
            verbose=False,
        )
        if success:
            cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
            if os.path.exists(cache_path):
                return np.load(cache_path)
    # print('end')
    # quit()

    matrix = get_attention_matrix(text, model_name, layer, head, device_map)

    if sentences is None:
        return matrix

    sentence_boundaries = get_sentence_token_boundaries(text, sentences, model_name)

    result = _compute_averaged_matrix(matrix, sentence_boundaries)

    if cache_dir and text_id:
        cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, result)

    return result


def get_vertical_scores(
    avg_mat: np.ndarray,
    proximity_ignore: int = 20,
    control_depth: bool = True,
    score_type: str = "mean",
) -> np.ndarray:
    """
    Calculate vertical attention scores from an averaged attention matrix.

    Args:
        avg_mat: Averaged attention matrix
        proximity_ignore: Number of nearby tokens to ignore (default 20)
        control_depth: Whether to multiply by depth/position (default True)
        score_type: How to aggregate scores - "mean" or "median" (default "mean")

    Returns:
        Array of vertical scores
    """

    # Clean the matrix - set upper triangle to NaN
    n = avg_mat.shape[0]
    trius = np.triu_indices_from(avg_mat, k=1)

    avg_mat = avg_mat.copy()
    avg_mat[trius] = np.nan

    trils = np.triu_indices_from(
        avg_mat, k=-proximity_ignore + 1
    )  # has no effect if not subtracting avg
    avg_mat[trils] = np.nan

    if control_depth:
        per_row = np.sum(~np.isnan(avg_mat), axis=1)
        avg_mat = stats.rankdata(avg_mat, axis=1, nan_policy="omit") / per_row[:, None]

    n = avg_mat.shape[-1]
    vert_scores = []

    for i in range(n):
        vert_lines = avg_mat[i + proximity_ignore :, i]

        if score_type == "mean":
            if len(vert_lines) == 0:
                # prevents "RuntimeWarning: Mean of empty slice"
                vert_score = np.nan 
            else:
                vert_score = np.nanmean(vert_lines)
        elif score_type == "median":
            if len(vert_lines) == 0:
                vert_score = np.nan
            else:
                vert_score = np.nanmedian(vert_lines)
        else:
            raise ValueError(f"Unknown score_type: {score_type}")

        vert_scores.append(vert_score)

    return np.array(vert_scores)


def get_attention_to_step(
    text: str,
    model_name: str,
    layer: int,
    head: int,
    step_idx: int,
    sentences: List[str],
    device_map: str = "auto",
    cache_dir: Optional[str] = "attn_cache",
) -> np.ndarray:
    """
    Get attention from all tokens to a specific step/sentence.

    Args:
        text: Input text to analyze
        model_name: Name of the model
        layer: Layer index
        head: Head index
        step_idx: Index of the target step/sentence
        sentences: List of sentences for chunking
        device_map: Device mapping for model loading
        cache_dir: Directory to cache matrices

    Returns:
        Array of attention weights to the target step
    """
    avg_matrix = get_avg_attention_matrix(
        text,
        model_name,
        layer,
        head,
        sentences,
        device_map,
        cache_dir=cache_dir,
    )

    return avg_matrix[:, step_idx]
