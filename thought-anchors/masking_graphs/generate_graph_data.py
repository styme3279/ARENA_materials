#!/usr/bin/env python
"""
Prepare and cache SuppObj instances for later visualization.
This script runs the necessary analysis and saves the results.
"""

import os
import sys
import asyncio
import shutil
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
import random
from pathlib import Path


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import diag_multi_k

from analyze_thinking import get_combined_df
from base_responses_thinking import get_thinking_prompts
from resample.fireworks_logprobs import gen_with_config

from resample.provider_config import ProviderConfig
from resample.supp import SuppObj, check_if_SO_exists
from resample.rate_limiter import set_rate_limit
import hashlib
from pkld import pkld
from resample.map_to_openrouter import get_openrouter_model_name
from resample.plot_suppression import (
    plot_k_profiles_by_domain,
    create_accuracy_vs_kl_scatter,
    create_k_correlation_timeseries,
)


async def prep_qwen_thinking_suppobj(
    prompt,
    max_tokens=0,
    model="qwen3-30b-a3b",
    verbose=False,
    req_SO_exist=False,
    clean_matrix=False,
    log_ratio=False,
):
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    if check_if_SO_exists(
        "qwen_mmlu", prompt_hash, max_tokens, model, clean_matrix=clean_matrix
    ):
        print(
            f"SuppObj already exists for {prompt_hash} with {max_tokens} tokens and model {model}"
        )
        SO = SuppObj.load(
            "qwen_mmlu",
            prompt_hash,
            max_tokens,
            model=model,
            clean_matrix=clean_matrix,
        )
        mean_kl, total_kl, first3_mean_kl = (
            SO.mean_kl_matrix,
            SO.total_kl_matrix,
            SO.first3_mean_kl_matrix,
        )
        return mean_kl, total_kl, first3_mean_kl
    if req_SO_exist:
        print(
            f"SuppObj does not exist for {prompt_hash} with {max_tokens} tokens and model {model}"
        )
        return None, None, None

    cfg = ProviderConfig(model=model, max_tokens=max_tokens)

    base_rollout = await gen_with_config(
        prompt, 1, config=cfg, verbose=verbose, model=model
    )
    base_resp = base_rollout.responses[0]

    non_thinking_models = {"llama-v3p3-70b-instruct"}
    if model in non_thinking_models:
        assert (
            "<think>" not in base_resp.full_text
        ), f"<think> found in non-thinking model: {base_resp.full_text=}"
    else:
        assert (
            "<think>" in base_resp.full_text
        ), f"No <think> in {base_resp.full_text=}"
        cnt_think = base_resp.full_text.count("<think>")
        assert cnt_think == 1, f"Too many <think> in {base_resp.full_text=}"
    if "<think>" not in base_resp.full_text:
        assert (
            "Let's think step by step" in base_resp.full_text
        ), f"No <think> or 'Let's think step by step' in {base_resp.full_text}"
        assert (
            "final answer:" in base_resp.full_text.lower()
        ), f"No 'final answer:' in {base_resp.full_text}"
        assert (
            "<|start_header_id|>assistant<|end_header_id|>\n"
            in base_resp.full_text
        ), f"No '<|start_header_id|>assistant<|end_header_id|>\n' in {base_resp.full_text}"
        base_resp.full_text = base_resp.full_text.replace(
            "</think>", ""
        )  # sometimes my code adds a </think> at the end by default
        header_str = "<|start_header_id|>assistant<|end_header_id|>\n"
        idx_header = base_resp.full_text.lower().index(header_str)
        idx_header = idx_header + len(header_str)
        # this makes non-thinking prompts compatible with my code designed for thinking models
        base_resp.full_text = (
            base_resp.full_text[:idx_header]
            + "<think>"
            + base_resp.full_text[idx_header:]
        )

        idx_final_answer = base_resp.full_text.lower().rindex("final answer:")
        base_resp.full_text = (
            base_resp.full_text[:idx_final_answer]
            + "</think>"
            + base_resp.full_text[idx_final_answer:]
        )

    SO = SuppObj(base_response=base_resp, do_paragraphs=False, model_name=model)
    SO.split_into_sentences()
    SO.prep_supp_prompts()

    # Run suppression rollouts - rate limiting is handled globally
    await SO.run_supp_rollouts_async(model=model, cfg=cfg, verbose=verbose)
    print("Making KL matrix")

    mean_kl, total_kl, first3_mean_kl = SO.run_supp_KL_matrix(
        clean_matrix=clean_matrix, log_ratio=log_ratio
    )

    # Save the SuppObj
    SO.save(
        "qwen_mmlu",
        prompt_hash,
        max_tokens,
        model=model,
        clean_matrix=clean_matrix,
    )
    return mean_kl, total_kl, first3_mean_kl


@pkld(overwrite=False)
async def async_main(
    prompts,
    model,
    max_concurrent=10,
    req_SO_exist=False,
    clean_matrix=False,
    log_ratio=False,
):
    """Async version of main for proper event loop handling

    Args:
        prompts: List of prompts to process
        max_concurrent: Number of prompts to process concurrently

    Note: Rate limiting is handled globally via the RateLimiter class
    """
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm

    # Configuration variables
    print(
        f"Processing {len(prompts)} prompts with max {max_concurrent} concurrent prompt operations"
    )
    print(f"Rate limiting is handled globally at the API level")

    # Create semaphore for controlling concurrent prompt processing
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create outer progress bar for overall prompt completion (yellowish orange)
    overall_pbar = tqdm(
        total=len(prompts),
        desc="Overall Progress",
        position=0,
        leave=True,
        colour="yellow",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    # Track completed prompts
    completed = []
    completed_lock = asyncio.Lock()

    async def process_with_semaphore(prompt, idx):
        """Process a single prompt with semaphore rate limiting"""
        async with semaphore:

            # try:
            result = await prep_qwen_thinking_suppobj(
                prompt,
                max_tokens=0,
                model=model,
                verbose=False,
                req_SO_exist=req_SO_exist,
                clean_matrix=clean_matrix,
                log_ratio=log_ratio,
            )

            # Update overall progress bar
            async with completed_lock:
                completed.append((idx, result))
                overall_pbar.update(1)

            return idx, result

    # prompts = prompts[0:1]
    tasks = [
        process_with_semaphore(prompt, i) for i, prompt in enumerate(prompts)
    ]
    # Run all tasks concurrently with proper error handling
    print("Starting concurrent processing...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Close the progress bar
    overall_pbar.close()

    # Process results
    successful_count = 0
    error_count = 0
    not_done_count = 0
    results_idxs = []
    results_data = []

    # Process results - they should already be (idx, result) tuples
    for i, result in enumerate(results):
        if result is None:
            continue
        if result[1] is None:
            continue
        if req_SO_exist:
            if result[1][0] is None:
                not_done_count += 1
                continue
        if isinstance(result, Exception):
            print(f"Task failed with exception: {result}")
            error_count += 1
        elif isinstance(result, tuple):
            idx, data = result
            if data is None:
                print(f"Task {idx+1} returned None (likely handled error)")
                error_count += 1
            else:
                mean_kl, total_kl, first3_mean_kl = data
                print(f"Task {idx+1} completed successfully:")
                print(f"  Mean KL shape: {mean_kl.shape}")
                print(f"  Total KL shape: {total_kl.shape}")
                print(f"  First3 Mean KL shape: {first3_mean_kl.shape}")
                successful_count += 1
                results_idxs.append(i)
                results_data.append(data)
        else:
            print(f"Unexpected result type: {type(result)}")
            error_count += 1

    print(f"\nSummary:")
    print(f"  Successful: {successful_count}")
    print(f"  Errors: {error_count}")
    if req_SO_exist:
        print(f"  Not done: {not_done_count}")
    print(f"  Total: {len(prompts)}")

    return results_idxs, results_data


def print_graph_descriptive_stats(graphs):
    num_sentences = [graph.shape[0] for graph in graphs]
    print(f"Mean sentence length: {np.mean(num_sentences):.1f}")
    print(f"\tMedian sentence length: {np.median(num_sentences):.1f}")
    print(f"\tMax sentence length: {np.max(num_sentences):.1f}")
    print(f"\tMin sentence length: {np.min(num_sentences):.1f}")
    print(f"\tStd sentence length: {np.std(num_sentences):.1f}")
    SE_sentence = np.std(num_sentences) / np.sqrt(len(num_sentences))
    print(f"SE sentence length: {SE_sentence:.1f}")
    print(f"\t95% CI sentence length: {1.96 * SE_sentence:.1f}")


def add_question_stats_to_graph_df(
    graph_df,
    results_idxs,
    model,
    max_questions,
    max_nonthinking_accuracy,
    q_data_list,
    prompts,
):
    q_data_list = [q_data_list[i] for i in results_idxs]
    prompts = [prompts[i] for i in results_idxs]
    q_hashes = [q_data["hash_key"] for q_data in q_data_list]

    model2base = {
        r"meta-llama/llama-3.3-70b-instruct": "llama-v3p3-70b-instruct",
        r"deepseek/deepseek-r1-distill-llama-70b": "llama-v3p3-70b-instruct",
        r"qwen3-30b-a3b": "qwen3-30b-a3b",
        "llama-v3p3-70b-instruct": "llama-v3p3-70b-instruct",
    }

    combined_df = pkld(get_combined_df, overwrite=False)(
        model2base[model], model, max_questions, max_nonthinking_accuracy
    )
    combined_df = combined_df[combined_df["hash_key"].isin(q_hashes)]
    combined_df.set_index("hash_key", inplace=True)
    combined_df = combined_df.loc[q_hashes]
    assert len(graph_df) == len(combined_df)
    graph_df["hash_key"] = q_hashes
    graph_df.set_index("hash_key", inplace=True)

    combined_df = pd.concat([combined_df, graph_df], axis=1)

    return combined_df


def get_graph_fp(model, require_correct):
    model_name = model.replace("/", "-")

    # Create model-specific output directory
    model_dir = f"results/graph_data/{model_name}"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    if require_correct:
        fn = "df_correct.csv"
    else:
        fn = "df_incorrect.csv"

    fp_out = f"{model_dir}/{fn}"

    return fp_out, model_name


def main(
    model,
    get_shortest=True,
    max_questions=1000,
    require_correct=True,
    max_concurrent=10,  # Number of prompts to process concurrently
    rpm_limit=250,  # API rate limit in requests per minute
    max_nonthinking_accuracy=0.5,
    req_SO_exist=False,
    clean_matrix=False,
    log_ratio=False,
):
    """Synchronous main function that runs the async version

    Args:
        get_shortest: Whether to get shortest prompts first
        max_questions: Maximum number of questions to process
        require_correct: Whether to require correct answers
        max_concurrent: Number of prompts to process concurrently (doesn't affect rate limit)
        rpm_limit: API rate limit in requests per minute (default: 550)
        max_nonthinking_accuracy: Maximum accuracy for non-thinking prompts
    """

    if require_correct is None:
        get_shortest = False
    set_rate_limit(rpm_limit)

    model_OR = get_openrouter_model_name(model)

    print("Loading thinking prompts...")
    prompts, q_data_list = get_thinking_prompts(
        model=model_OR,
        max_questions=max_questions,
        get_shortest=get_shortest,
        require_correct=require_correct,
        max_nonthinking_accuracy=max_nonthinking_accuracy,
    )

    assert len(prompts) == len(q_data_list)
    combined_list = list(zip(q_data_list, prompts))
    sorted_combined = sorted(combined_list, key=lambda x: x[1])
    q_data_list, prompts = zip(*sorted_combined)

    print(f"{len(prompts)=}")
    # quit()

    # prompts = prompts[:100]
    # q_data_list = q_data_list[:100]
    # prompts = prompts[::-1]
    # q_data_list = q_data_list[::-1]

    results_idxs, results_data = asyncio.run(
        async_main(
            prompts,
            model=model,
            max_concurrent=max_concurrent,
            req_SO_exist=req_SO_exist,
            clean_matrix=clean_matrix,
            log_ratio=log_ratio,
        )
    )
    # print(f"{len(results_idxs)=}")
    # quit()

    assert len(results_idxs) == len(results_data)
    print(f"{len(results_idxs)=}")
    graphs = [result[0] for result in results_data]
    print_graph_descriptive_stats(graphs)

    # graph_vars = [get_graph_vars(graph) for graph in tqdm(graphs, desc="Computing graph vars")]
    graph_vars = [
        get_distance_strengths(graph, max_k=128)
        for graph in tqdm(graphs, desc="Computing distance strengths")
    ]
    graph_df = pd.DataFrame(graph_vars)
    graph_df["num_sentences"] = [graph.shape[0] for graph in graphs]
    graph_vars = list(graph_df.columns)

    df = add_question_stats_to_graph_df(
        graph_df=graph_df,
        results_idxs=results_idxs,
        model=model,
        max_questions=max_questions,
        max_nonthinking_accuracy=max_nonthinking_accuracy,
        q_data_list=q_data_list,
        prompts=prompts,
    )
    print(f"{len(df)=}")

    df_cols = set(df.columns)
    graph_df_cols = set(graph_df.columns)
    assert graph_df_cols.issubset(df_cols)
    assert len(df) == len(graph_df) == len(results_data)

    fp_out, model_name = get_graph_fp(model, require_correct)
    print(f"Saving df ({len(df)=}) to: {fp_out}")
    df.to_csv(fp_out)

    # # Visualize k-profiles by subject/domain
    # plot_k_profiles_by_domain(df, model_name=model_name)

    # # Create accuracy vs KL scatter plots and interactive HTML
    # create_accuracy_vs_kl_scatter(df, model_name=model_name)

    # # Create time series plot of correlations by k
    # # You can customize which variables to correlate with by passing correlation_targets
    # # Options include: 'thinking_improvement', 'thinking_is_correct', 'prob_correct', etc.
    # create_k_correlation_timeseries(
    #     df,
    #     max_k=50,
    #     correlation_targets=[
    #         "thinking_improvement",
    #         "thinking_is_correct",
    #         "prob_correct",
    #     ],
    #     model_name=model,
    # )


def get_distance_strengths(
    total_kl, max_k, buffer_k=0, include_count=False, limit_half=True
):
    out = {}
    if limit_half:
        max_k = min(max_k, total_kl.shape[0] // 2)
    else:
        max_k = min(max_k, total_kl.shape[0])

    for k in range(1, max_k):
        if buffer_k > 0:
            k_start = k - buffer_k
            k_end = k + buffer_k
            vals = diag_multi_k(total_kl, k_start=k_start, k_end=k_end)
            val = np.nanmean(vals)
        else:
            vals = np.diag(total_kl, k=k)
            val = np.nanmean(vals)

        out[f"total_kl_k{k}"] = val
        if include_count:
            out[f"count_diag_{k}"] = len(vals)
    # grand_mean = np.nanmean(list(out.values()))
    # for k in range(1, max_k):
    #     out[f"total_kl_k{k}"] = out[f"total_kl_k{k}"] - grand_mean
    return out


# @pkld
def get_graph_vars(total_kl, k=8):
    """
    Compute comprehensive graph statistics from KL divergence matrix.
    Uses the new graph_funcs module for detailed analysis.
    """
    # Import here to avoid circular dependencies
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from graph_funcs import get_thought_graph_features, get_key_metrics

    # Prepare matrix for graph analysis (set lower triangle to NaN)
    matrix = total_kl.copy()
    triu_indices = np.tril_indices_from(
        matrix, k=-1
    )  # Everything below diagonal
    matrix[triu_indices] = np.nan

    triu_indices = np.tril_indices_from(total_kl, k=k)
    total_kl[triu_indices] = 0

    # Get comprehensive features
    try:
        # Get all features for thorough analysis
        all_features = get_thought_graph_features(matrix)

        # If there's an error in the features, return a simple fallback
        if "error" in all_features:
            # Fallback to simple statistics
            triu_indices = np.tril_indices_from(total_kl, k=k)
            total_kl[triu_indices] = np.nan
            triu_indices_wider = np.tril_indices_from(total_kl, k=k * 4)
            total_kl_wider = total_kl[triu_indices_wider]
            total_kl_mean = np.nanmean(total_kl_wider)
            return {"total_kl_medium": total_kl_mean}

        # Add the original simple metric for backward compatibility
        triu_indices_simple = np.tril_indices_from(total_kl, k=k)
        total_kl_simple = total_kl.copy()
        total_kl_simple[triu_indices_simple] = np.nan
        triu_indices_wider = np.tril_indices_from(total_kl_simple, k=k * 4)
        total_kl_wider = total_kl_simple[triu_indices_wider]
        all_features["total_kl_medium"] = float(np.nanmean(total_kl_wider))

        return all_features

    except Exception as e:
        print(f"Warning: Error computing graph features: {e}")
        # Return simple fallback
        triu_indices = np.tril_indices_from(total_kl, k=k)
        total_kl[triu_indices] = np.nan
        triu_indices_wider = np.tril_indices_from(total_kl, k=k * 4)
        total_kl_wider = total_kl[triu_indices_wider]
        total_kl_mean = np.nanmean(total_kl_wider)
        return {"total_kl_medium": total_kl_mean}


if __name__ == "__main__":
    REQ_SO_EXIST = False
    MODEL = "qwen3-30b-a3b"
    LOG_RATIO = False
    REQUIRE_CORRECT = True
    # REQUIRE_CORRECT = False
    # REQ_SO_EXIST = True

    main(
        model=MODEL,
        max_questions=16000,
        max_nonthinking_accuracy=0.5,
        log_ratio=LOG_RATIO,
        req_SO_exist=REQ_SO_EXIST,
        require_correct=REQUIRE_CORRECT,
    )
    # main(max_questions=16000, max_nonthinking_accuracy=None)
