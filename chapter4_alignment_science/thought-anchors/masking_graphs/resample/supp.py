"""
Dataclasses for structured LLM response data.

This module provides type-safe dataclass representations of the response
data from LLM APIs, particularly for use with logprobs and caching.
"""

from asyncio import Task, run
import asyncio
from dataclasses import dataclass, field, asdict
from pprint import pprint
from typing import List, Dict, Optional, Any, Union, Tuple
import numpy as np
from tqdm import tqdm
import pickle
import os

from resample.fireworks_logprobs import gen_with_config
from resample.kl_funcs import kl_divergence_from_logprobs_simple
from resample.provider_config import ProviderConfig
from resample.sentence_splitter import split_into_paragraphs_safe, string_to_sentences
from resample.rollouts import FwResponse, Rollouts
from pathlib import Path

from utils import split_solution_keep_spacing


def get_st_stop_think_str(model="qwen3-30b-a3b"):
    if "gpt-oss" in model:
        return "<|channel|>analysis<|message|>", "<|end|>"
    else:
        return "<think>", "</think>"


@dataclass
class SuppObj:
    base_response: FwResponse
    do_paragraphs: bool = False  # TODO: test
    model_name: str = "qwen3-30b-a3b"  # Store model name for think token handling
    sentences: List[str] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)
    supp_prompts: List[str] = field(default_factory=list)
    supp_responses: List[FwResponse] = field(default_factory=list)

    def split_into_sentences(self) -> List[str]:
        """
        Note, for do_paragraphs
        """
        if self.do_paragraphs:
            raise ValueError("do_paragraphs is not supported yet")
            # paragraphs, paragraph_positions = split_into_paragraphs_safe(
            #     self.base_response.full_text
            # )
            # sentences, positions = paragraphs, paragraph_positions
        else:
            # print('test')
            sentences, positions = split_solution_keep_spacing(
                self.base_response.full_text, get_idxs=True, drop_think=True
            )

        # sentences, positions = string_to_sentences(self.base_response.full_text)

        # Get model-specific think tokens
        think_open, think_close = get_st_stop_think_str(self.model_name)
        think_str = f"\n{think_open}\n" if self.do_paragraphs else think_open
        post_think_str = f"\n{think_close}" if self.do_paragraphs else None

        sentences_new = []
        positions_new = []
        has_think = False
        has_post_think = False
        has_think = True
        has_post_think = True
        for idx, sentence in enumerate(sentences):
            if think_str in sentence and False:
                sentence_pre, sentence_post = sentence.split(think_str)
                sentences_new.append(sentence_pre)
                positions_new.append(positions[idx])
                sentences_new.append(think_str)
                pos_think = positions[idx] + len(sentence_pre)
                positions_new.append(pos_think)
                sentences_new.append(sentence_post)
                pos_new = pos_think + len(think_str)
                positions_new.append(pos_new)
                # pos_new
                has_think = True
            elif post_think_str is not None and post_think_str in sentence:
                sentence_pre, sentence_post = sentence.split(post_think_str)
                sentences_new.append(sentence_pre)
                positions_new.append(positions[idx])
                sentences_new.append(post_think_str)
                pos_new = positions[idx] + len(sentence_pre)
                positions_new.append(pos_new)
                sentences_new.append(sentence_post)
                pos_new = pos_new + len(post_think_str)
                positions_new.append(pos_new)
                has_post_think = True
            else:
                sentences_new.append(sentence)
                positions_new.append(positions[idx])
        if not has_think:
            think_open, _ = get_st_stop_think_str(self.model_name)
            raise ValueError(f"No {think_open} found in the response: {sentences=}")

        if post_think_str is not None:
            if not has_post_think:
                _, think_close = get_st_stop_think_str(self.model_name)
                raise ValueError(f"No {think_close} found in the response: {sentences=}")

        self.sentences = sentences_new
        self.positions = positions_new

        for sentence, pos in zip(sentences_new, positions_new):
            if sentence == think_str:
                continue
            if False:  # breaks for Llama 3.1 8B
                assert (
                    self.base_response.full_text[pos : pos + len(sentence)] == sentence
                ), f"{self.base_response.full_text[pos : pos + len(sentence)]=} != {sentence=}"
                # num_cnt = self.base_response.full_text.count(sentence)

    def prep_supp_prompts(self) -> List[str]:
        """
        Create a list of prompts where each prompt is the full text with one sentence omitted.
        Uses positions to handle duplicate sentences correctly.

        Returns:
            List of prompts, one for each sentence, with that sentence removed
        """
        assert self.sentences, "self.sentences is empty. run split_into_sentences() first."
        assert self.positions, "self.positions is empty. run split_into_sentences() first."

        full_text = self.base_response.full_text
        prompts = []

        for i, (sentence, pos) in enumerate(zip(self.sentences, self.positions)):
            # For each sentence, create a prompt with that sentence removed
            # Use the position to extract exactly the right instance

            # Find the end position of this sentence
            if i < len(self.positions) - 1:
                # Not the last sentence - use the next sentence's position
                end_pos = self.positions[i + 1]
            else:
                # Last sentence - use the position plus sentence length
                end_pos = pos + len(sentence)

            # Create the prompt by removing this exact sentence instance
            # Part before the sentence + part after the sentence
            prompt_without_sentence = full_text[:pos] + full_text[end_pos:]
            prompts.append(prompt_without_sentence)

        # Assert that we have the same number of prompts as sentences
        assert len(prompts) == len(
            self.sentences
        ), f"Number of prompts ({len(prompts)}) doesn't match number of sentences ({len(self.sentences)})"

        # Store for later use if needed
        self.supp_prompts = prompts

        return prompts

    async def run_supp_rollouts_async(
        self, model="qwen3-30b-a3b", cfg: ProviderConfig = None, verbose=False, **kwargs
    ) -> List[Rollouts]:
        """
        Async version of run_supp_rollouts for use within async contexts.
        Rate limiting is handled globally by the RateLimiter class.

        Args:
            model: Model to use for generation
            cfg: ProviderConfig for generation settings
            **kwargs: Additional arguments for ProviderConfig
        """
        assert self.supp_prompts, "self.supp_prompts is empty. run prep_supp_prompts() first."
        if cfg is None:
            cfg = ProviderConfig(model=model, max_tokens=0, **kwargs)

        # Create tasks wrapped with their indices
        async def run_with_index(prompt, index):
            """Wrapper to include index with result."""
            result = await gen_with_config(prompt, num_responses=1, config=cfg, verbose=verbose)
            return index, result

        # Create tasks with indices embedded
        tasks = [
            asyncio.create_task(run_with_index(prompt, i))
            for i, prompt in enumerate(self.supp_prompts)
        ]

        # Initialize results list with None placeholders
        results = [None] * len(tasks)

        # Create progress bar
        with tqdm(total=len(tasks), desc="Running suppression rollouts") as pbar:
            # Use as_completed to update progress bar as tasks finish
            for task in asyncio.as_completed(tasks):
                idx, result = await task
                results[idx] = result
                pbar.update(1)

        responses = []
        for rollout in results:
            responses.append(rollout.responses[0])
        self.supp_responses = responses
        return responses

    def run_supp_rollouts(self, model="qwen3-30b-a3b", **kwargs) -> List[Rollouts]:
        """
        Run generation for all suppression prompts asynchronously with progress bar.

        This creates multiple coroutines and runs them concurrently, updating a progress
        bar as each task completes.
        """
        # Use nest_asyncio to allow nested event loops
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            pass

        # Run the async version
        return asyncio.run(self.run_supp_rollouts_async(model=model, **kwargs))

    def calculate_kl_divergence(
        self, base_logprobs: List[float], supp_logprobs: List[float]
    ) -> float:
        """Calculate KL divergence between two log probability distributions.

        KL(base||supp) = Σ exp(base_logprob) * (base_logprob - supp_logprob)
        """
        kl_div = 0.0
        for base_lp, supp_lp in zip(base_logprobs, supp_logprobs):
            # Convert log prob to prob for weighting
            prob = np.exp(base_lp)
            kl_div += prob * (base_lp - supp_lp)
        return kl_div

    def align_tokens_with_hole(
        self,
        base_tokens: List[str],
        base_offsets: List[int],
        supp_tokens: List[str],
        supp_offsets: List[int],
        hole_start: int,
        hole_end: int,
    ) -> Tuple[List[int], List[int], int, int]:
        """Align tokens between base and suppression responses, accounting for the removed sentence.

        Returns:
            base_indices: Indices of base tokens that align
            supp_indices: Indices of suppression tokens that align
            hole_start_token_idx: Index in base tokens where the hole starts
            hole_end_token_idx: Index in base tokens where the hole ends
        """
        base_indices = []
        supp_indices = []
        hole_start_token_idx = -1
        hole_end_token_idx = -1

        # Find where the hole starts and ends in token space
        for i, offset in enumerate(base_offsets):
            if hole_start_token_idx == -1 and offset >= hole_start:
                # Check if the previous token actually contains the hole_start position
                if i > 0 and base_offsets[i - 1] < hole_start and offset > hole_start:
                    # The sentence boundary is in the MIDDLE of token i-1
                    # So token i-1 should be included in the hole
                    hole_start_token_idx = i - 1
                else:
                    hole_start_token_idx = i
            if offset >= hole_end:
                hole_end_token_idx = i
                break

        # hole_end_token_idx += 5
        if (
            base_tokens[hole_end_token_idx - 1]
            in [
                ".\n",
                ".\n\n",
                "?\n",
                "?\n\n",
                "!\n",
                "!\n\n",
                "\n\n",
                "\n",
                " \n\n",
                " \n",
            ]
            and hole_end_token_idx >= 0
        ):
            hole_end_token_idx += 2
        elif base_tokens[hole_end_token_idx - 1] in [":\n", ":\n\n"]:
            hole_end_token_idx += 3

        # If we didn't find end, it's at the end of tokens
        if hole_end_token_idx == -1:
            hole_end_token_idx = len(base_tokens)

        # PART 1: Align tokens BEFORE the hole
        # The TEXT should be identical, but tokenization might differ
        tokens_before_hole = (
            hole_start_token_idx if hole_start_token_idx > 0 else 0
        )  # Fix: should be 0 not len(base_tokens)
        # tokens_before_hole -= 1

        # Build the expected text before the hole
        expected_text_before_hole = (
            "".join(base_tokens[:tokens_before_hole]) if tokens_before_hole > 0 else ""
        )

        # Only try to align if there are tokens before the hole
        if tokens_before_hole > 0:
            # Find how many supp tokens cover the same text
            supp_text = ""
            supp_tokens_needed = 0
            for i, token in enumerate(supp_tokens):
                supp_text += token
                supp_tokens_needed = i + 1
                if supp_text == expected_text_before_hole:
                    # Perfect match
                    break
                elif len(supp_text) >= len(expected_text_before_hole):
                    # We've gone past - tokenization is different
                    # Adjust to not include the last token that went over
                    supp_tokens_needed -= 1
                    break

            # Now align token by token ONLY where they exactly match
            for i in range(min(tokens_before_hole, supp_tokens_needed, len(supp_tokens))):
                if i < len(base_tokens) and i < len(supp_tokens):
                    if base_tokens[i] == supp_tokens[i]:
                        base_indices.append(i)
                        supp_indices.append(i)
                    # Don't add mismatched tokens!

        # PART 2: Find where to resume alignment after the hole
        if hole_end_token_idx < len(base_tokens):
            # We need to find where in supp_tokens we can resume
            # The supp tokens should continue from roughly where we left off

            # Start searching from where we stopped before the hole
            supp_search_start = len(supp_indices) if supp_indices else 0

            # Look for the first matching sequence after the hole
            best_match_supp_idx = -1
            best_match_score = 0

            for supp_idx in range(supp_search_start, len(supp_tokens)):
                # Check how many consecutive tokens match
                match_count = 0
                max_check = min(
                    10, len(base_tokens) - hole_end_token_idx, len(supp_tokens) - supp_idx
                )

                for j in range(max_check):
                    if base_tokens[hole_end_token_idx + j] == supp_tokens[supp_idx + j]:
                        match_count += 1
                    else:
                        break

                # If this is a better match, use it
                if match_count > best_match_score:
                    best_match_score = match_count
                    best_match_supp_idx = supp_idx

                    # If we found a perfect match of at least 3 tokens, use it
                    if match_count >= 5:
                        break

            # If we found a good alignment point, align the rest
            if best_match_supp_idx >= 0 and best_match_score >= 5:
                # Align all remaining tokens
                remaining_base = len(base_tokens) - hole_end_token_idx
                remaining_supp = len(supp_tokens) - best_match_supp_idx
                tokens_to_align = min(remaining_base, remaining_supp)

                aligned_count = 0
                for j in range(tokens_to_align):
                    base_idx = hole_end_token_idx + j
                    supp_idx = best_match_supp_idx + j
                    if "\n" in base_tokens[base_idx]:
                        continue
                    if "\n" in supp_tokens[supp_idx]:
                        continue

                    # Only align if tokens match
                    if base_tokens[base_idx] == supp_tokens[supp_idx]:
                        base_indices.append(base_idx)
                        supp_indices.append(supp_idx)
                        aligned_count += 1
                    else:
                        # Stop aligning if we hit a mismatch
                        break

        return base_indices, supp_indices, hole_start_token_idx, hole_end_token_idx

    def find_sentence_token_ranges(self) -> List[Tuple[int, int]]:
        """Find the token index ranges for each sentence in the base response.

        Returns:
            List of (start_token_idx, end_token_idx) for each sentence
        """
        base_offsets = self.base_response.logprobs.text_offset
        sentence_ranges = []

        for i, (sentence, position) in enumerate(zip(self.sentences, self.positions)):
            # Find sentence end position
            if i < len(self.positions) - 1:
                end_pos = self.positions[i + 1]
            else:
                end_pos = position + len(sentence)

            # Find token indices for this sentence
            start_token_idx = -1
            end_token_idx = -1

            for j, offset in enumerate(base_offsets):
                # Find start token (including tokens that span the boundary)
                if start_token_idx == -1 and offset >= position:
                    if j > 0 and base_offsets[j - 1] < position and offset > position:
                        # Sentence starts in middle of token j-1
                        start_token_idx = j - 1
                    else:
                        start_token_idx = j

                # Find end token
                if offset >= end_pos:
                    end_token_idx = j
                    break

            if end_token_idx == -1:
                end_token_idx = len(base_offsets)

            sentence_ranges.append((start_token_idx, end_token_idx))

        return sentence_ranges

    def run_supp_KL(self) -> List[Dict[str, Any]]:
        """Calculate KL divergence for each suppression response relative to base.

        Returns list of dicts with:
        - sentence_idx: Index of removed sentence
        - sentence: The removed sentence text
        - kl_before_hole: KL divergence before the hole (should be ~0)
        - kl_after_hole: KL divergence after the hole
        - total_kl: Total KL divergence
        - num_aligned_tokens: Number of tokens that were aligned
        """
        assert self.supp_responses, "self.supp_responses is empty. run run_supp_rollouts() first."
        assert self.base_response.logprobs is not None, "Base response needs logprobs enabled"

        results = []

        base_logprobs = self.base_response.logprobs
        base_tokens = base_logprobs.tokens
        base_token_logprobs = base_logprobs.token_logprobs
        base_offsets = base_logprobs.text_offset
        self.holes_start_char = []
        self.holes_end_char = []
        self.holes_start_token = []
        self.holes_end_token = []

        for idx, (supp_resp, sentence, position) in enumerate(
            zip(self.supp_responses, self.sentences, self.positions)
        ):
            if supp_resp.logprobs is None:
                results.append(
                    {
                        "sentence_idx": idx,
                        "sentence": sentence,
                        "error": "No logprobs in suppression response",
                    }
                )
                continue

            supp_logprobs = supp_resp.logprobs
            supp_tokens = supp_logprobs.tokens
            supp_token_logprobs = supp_logprobs.token_logprobs
            supp_offsets = supp_logprobs.text_offset

            # Find the hole boundaries
            hole_start = position
            if idx < len(self.positions) - 1:
                hole_end = self.positions[idx + 1]
            else:
                hole_end = position + len(sentence)

            base_idx, supp_idx, hole_start_tok_idx, hole_end_tok_idx = self.align_tokens_with_hole(
                base_tokens, base_offsets, supp_tokens, supp_offsets, hole_start, hole_end
            )

            self.holes_start_char.append(hole_start)
            self.holes_end_char.append(hole_end)
            self.holes_start_token.append(hole_start_tok_idx)
            self.holes_end_token.append(hole_end_tok_idx)

            # Calculate maximum possible aligned tokens
            # This is total base tokens minus the tokens in the hole
            tokens_in_hole = hole_end_tok_idx - hole_start_tok_idx if hole_start_tok_idx >= 0 else 0
            max_possible_aligned = len(base_tokens) - tokens_in_hole

            if not base_idx:
                results.append(
                    {
                        "sentence_idx": idx,
                        "sentence": sentence,
                        "error": "Could not align tokens",
                        "max_possible_aligned": max_possible_aligned,
                    }
                )
                continue

            # Calculate KL divergence before and after hole
            kl_before = 0.0
            kl_after = 0.0
            tokens_before = 0
            tokens_after = 0

            # Only process truly aligned tokens
            actually_aligned = []

            for b_idx, s_idx in zip(base_idx, supp_idx):
                # STRICT CHECK: tokens must be exactly identical
                if base_tokens[b_idx] == supp_tokens[s_idx]:
                    actually_aligned.append((b_idx, s_idx))

            for b_idx, s_idx in actually_aligned:
                kl_value = np.exp(base_token_logprobs[b_idx]) * (
                    base_token_logprobs[b_idx] - supp_token_logprobs[s_idx]
                )

                # Use hole_start_tok_idx to determine if before or after
                if hole_start_tok_idx > 0 and b_idx < hole_start_tok_idx:
                    kl_before += kl_value
                    tokens_before += 1
                elif b_idx >= hole_end_tok_idx:
                    kl_after += kl_value
                    tokens_after += 1

            results.append(
                {
                    "sentence_idx": idx,
                    "sentence": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                    "position": position,
                    "kl_before_hole": kl_before,
                    "kl_after_hole": kl_after,
                    "total_kl": kl_before + kl_after,
                    "num_aligned_tokens": len(actually_aligned),
                    "max_possible_aligned": max_possible_aligned,
                    "alignment_ratio": (
                        len(actually_aligned) / max_possible_aligned
                        if max_possible_aligned > 0
                        else 0
                    ),
                    "tokens_before_hole": tokens_before,
                    "tokens_after_hole": tokens_after,
                    "hole_start_token": hole_start_tok_idx,
                    "hole_end_token": hole_end_tok_idx,
                }
            )

        self.kl_results = results
        return results

    def run_supp_KL_matrix(
        self, clean_matrix=True, log_ratio=False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate KL divergence matrices for sentence suppression effects.

        Returns three matrices (all num_sentences x num_sentences):
        1. mean_kl_matrix[i,j] = mean KL effect on sentence j when sentence i is removed
        2. total_kl_matrix[i,j] = total KL effect on sentence j when sentence i is removed
        3. first3_mean_kl_matrix[i,j] = mean KL for first 3 tokens of sentence j when i is removed

        Diagonal elements are NaN (can't measure effect on removed sentence).
        """
        assert self.supp_responses, "self.supp_responses is empty. run run_supp_rollouts() first."
        assert self.base_response.logprobs is not None, "Base response needs logprobs enabled"

        self.base_response = self.supp_responses[
            -1
        ]  # jank fix, don't know whats up with the tokenization in the base_response

        # Get base token info
        base_tokens = self.base_response.logprobs.tokens
        base_token_logprobs = self.base_response.logprobs.token_logprobs
        base_top_logprobs = self.base_response.logprobs.top_logprobs
        lowest_logprob = 0
        for d in base_top_logprobs:
            for key, v in d.items():
                if v is None:
                    assert key == "ERROR" or "<｜end▁of▁sentence｜>" in d, f"{d=}"
                    assert log_ratio, f"{log_ratio=} is False"
                    continue

                if v < lowest_logprob:
                    lowest_logprob = v

        base_offsets = self.base_response.logprobs.text_offset

        # Find token ranges for each sentence in base response
        sentence_ranges = self.find_sentence_token_ranges()

        # Initialize matrices
        num_sentences = min(len(self.sentences), len(self.supp_responses))
        mean_kl_matrix = np.full((num_sentences, num_sentences), np.nan)
        total_kl_matrix = np.full((num_sentences, num_sentences), np.nan)
        first3_mean_kl_matrix = np.full((num_sentences, num_sentences), np.nan)

        # For each removed sentence (hole)
        for hole_idx in range(num_sentences):
            supp_resp = self.supp_responses[hole_idx]

            # Handle both dict and dataclass responses
            if isinstance(supp_resp, dict):
                if supp_resp.get("logprobs") is None:
                    continue
                supp_tokens = supp_resp["logprobs"]["tokens"]
                supp_token_logprobs = supp_resp["logprobs"]["token_logprobs"]
                supp_top_logprobs = supp_resp["logprobs"]["top_logprobs"]
                supp_offsets = supp_resp["logprobs"]["text_offset"]
            else:
                if supp_resp.logprobs is None:
                    continue
                supp_tokens = supp_resp.logprobs.tokens
                supp_token_logprobs = supp_resp.logprobs.token_logprobs
                supp_top_logprobs = supp_resp.logprobs.top_logprobs
                supp_offsets = supp_resp.logprobs.text_offset

            # Get hole position

            hole_start = self.positions[hole_idx]
            if hole_idx < len(self.positions) - 1:
                hole_end = self.positions[hole_idx + 1]
            else:
                hole_end = hole_start + len(self.sentences[hole_idx])
            # hole_end = hole_end + 2

            # Align tokens between base and suppression
            base_idx, supp_idx, hole_start_tok_idx, hole_end_tok_idx = self.align_tokens_with_hole(
                base_tokens, base_offsets, supp_tokens, supp_offsets, hole_start, hole_end
            )

            if not base_idx:
                continue
            # Create mapping of aligned tokens
            token_alignment = {}
            for b_idx, s_idx in zip(base_idx, supp_idx):
                if base_tokens[b_idx] == supp_tokens[s_idx]:
                    token_alignment[b_idx] = s_idx

            # Calculate KL for each target sentence
            for target_idx in range(num_sentences):
                if target_idx == hole_idx:
                    continue  # Can't measure effect on removed sentence

                # Get token range for target sentence
                target_start, target_end = sentence_ranges[target_idx]

                # Adjust token range if target is after the hole
                if target_idx > hole_idx:
                    # Tokens shift because of the hole
                    # But we use the original indices from base response
                    pass  # We'll use the alignment mapping

                # Collect KL values for tokens in target sentence
                kl_values = []
                for token_idx in range(target_start, target_end):
                    if token_idx in token_alignment:
                        supp_token_idx = token_alignment[token_idx]
                        if log_ratio:
                            kl = (
                                base_token_logprobs[token_idx] - supp_token_logprobs[supp_token_idx]
                            )
                        else:
                            kl = kl_divergence_from_logprobs_simple(
                                base_top_logprobs[token_idx],
                                supp_top_logprobs[supp_token_idx],
                                missing_logprob=lowest_logprob,
                            )
                        kl_values.append(kl)

                if kl_values:
                    # Calculate metrics
                    mean_kl_matrix[hole_idx, target_idx] = np.mean(kl_values)
                    total_kl_matrix[hole_idx, target_idx] = np.sum(kl_values)

                    # First 3 tokens (or fewer if sentence is short)
                    first3_kl = kl_values[:3]
                    if first3_kl:
                        first3_mean_kl_matrix[hole_idx, target_idx] = np.mean(first3_kl)

        if clean_matrix:
            mean_kl_matrix[np.tril_indices_from(mean_kl_matrix)] = np.nan
            total_kl_matrix[np.tril_indices_from(total_kl_matrix)] = np.nan
            first3_mean_kl_matrix[np.tril_indices_from(first3_mean_kl_matrix)] = np.nan

            mean_kl_matrix_pad = mean_kl_matrix.copy()
            total_kl_matrix_pad = total_kl_matrix.copy()
            first3_mean_kl_matrix_pad = first3_mean_kl_matrix.copy()

            mean_kl_matrix -= np.nanmean(mean_kl_matrix_pad, axis=0, keepdims=True)
            total_kl_matrix -= np.nanmean(total_kl_matrix_pad, axis=0, keepdims=True)
            first3_mean_kl_matrix -= np.nanmean(first3_mean_kl_matrix_pad, axis=0, keepdims=True)

        self.mean_kl_matrix = mean_kl_matrix
        self.total_kl_matrix = total_kl_matrix
        self.first3_mean_kl_matrix = first3_mean_kl_matrix
        return mean_kl_matrix, total_kl_matrix, first3_mean_kl_matrix

    def save(
        self,
        key: str,
        prompt_hash: str,
        max_tokens: int,
        model: str = "qwen3-30b-a3b",
        clean_matrix=False,
    ):
        """Save the SuppObj to a pickle file."""

        cache_dir = os.path.join("cache", "SuppObj", model)
        clean_str = "" if clean_matrix else "_unclean_matrix"
        filename = f"{key}_tokens{max_tokens}/{prompt_hash}{clean_str}.pkl"
        filepath = os.path.join(cache_dir, filename)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save the object
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        print(f"Saved SuppObj to {filepath}")
        return filepath

    @staticmethod
    def load(
        key: str,
        prompt_hash: str,
        max_tokens: int,
        model: str = "qwen3-30b-a3b",
        clean_matrix=False,
    ) -> Optional["SuppObj"]:
        """Load a SuppObj from cache if it exists."""
        cache_dir = os.path.join("cache", "SuppObj", model)
        clean_str = "" if clean_matrix else "_unclean_matrix"
        filename = f"{key}_tokens{max_tokens}/{prompt_hash}{clean_str}.pkl"
        filepath = os.path.join(cache_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                obj = pickle.load(f)
            print(f"Loaded SuppObj from {filepath}")
            return obj
        else:
            return None


def check_if_SO_exists(
    key: str, prompt_hash: str, max_tokens: int, model: str = "qwen3-30b-a3b", clean_matrix=False
):
    cache_dir = os.path.join("cache", "SuppObj", model)
    clean_str = "" if clean_matrix else "_unclean_matrix"
    filename = f"{key}_tokens{max_tokens}/{prompt_hash}{clean_str}.pkl"
    filepath = os.path.join(cache_dir, filename)
    return os.path.exists(filepath)


def find_clipped_bounds(s0, s1):
    """
    Find the start and end indices in s0 where a substring was removed to create s1.

    Args:
        s0 (str): Original string
        s1 (str): String with a portion clipped from the middle of s0

    Returns:
        tuple: (start_index, end_index) where the substring was removed from s0
               Returns None if s1 is not a valid clipped version of s0
    """
    # Check if s1 could be a clipped version of s0
    if len(s1) > len(s0):
        return None

    # If strings are identical, no clipping occurred
    if s0 == s1:
        return None

    # Find the longest common prefix
    prefix_len = 0
    for i in range(min(len(s0), len(s1))):
        if s0[i] == s1[i]:
            prefix_len += 1
        else:
            break

    # Find the longest common suffix
    suffix_len = 0
    for i in range(1, min(len(s0), len(s1)) - prefix_len + 1):
        if s0[-i] == s1[-i]:
            suffix_len += 1
        else:
            break

    # Calculate the clipped region boundaries
    clip_start = prefix_len
    clip_end = len(s0) - suffix_len - 1

    # Verify that this is a valid clipping
    # The reconstructed string should match s1
    reconstructed = s0[:clip_start] + s0[clip_end + 1 :]
    if reconstructed != s1:
        return None

    return (clip_start, clip_end)


# def clean_logprobs()
