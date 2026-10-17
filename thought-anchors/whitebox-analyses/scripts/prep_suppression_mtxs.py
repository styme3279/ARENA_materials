import json
import os
import sys
from typing import Dict, List, Tuple, Optional
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import attention_analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.attn_supp_funcs import get_suppression_KL_matrix
from attention_analysis.receiver_head_funcs import (
    get_model_rollouts_root,
)


def get_all_problem_numbers(model_name="qwen-14b", include_incorrect=True):
    """
    Get all problem numbers from the rollouts directory.

    Args:
        model_name: Model name to get problems for
        include_incorrect: If True, include both correct and incorrect solutions

    Returns:
        List of (problem_num, is_correct) tuples
    """
    dir_root = get_model_rollouts_root(model_name)
    problem_list = []

    # Process correct solutions
    correct_dir = os.path.join(dir_root, "correct_base_solution")
    if os.path.exists(correct_dir):
        for problem_dir in os.listdir(correct_dir):
            if problem_dir.startswith("problem_"):
                problem_num = int(problem_dir.replace("problem_", ""))
                problem_list.append((problem_num, True))

    # Process incorrect solutions if requested
    if include_incorrect:
        incorrect_dir = os.path.join(dir_root, "incorrect_base_solution")
        if os.path.exists(incorrect_dir):
            for problem_dir in os.listdir(incorrect_dir):
                if problem_dir.startswith("problem_"):
                    problem_num = int(problem_dir.replace("problem_", ""))
                    problem_list.append((problem_num, False))

    return sorted(problem_list)


def process_all_problems_kl(
    model_name: str = "qwen-14b",
    p_nucleus: float = 0.9999,
    only_first: Optional[int] = None,
    take_log: bool = True,
    save_results: bool = True,
    output_dir: str = "kl_results",
    max_problems: Optional[int] = None,
    include_incorrect: bool = True,
) -> Dict[Tuple[int, bool], np.ndarray]:
    """
    Process all problems to compute sentence-to-sentence KL divergences.

    Args:
        model_name: Model to use
        p_nucleus: Nucleus sampling parameter
        only_first: Only process first N tokens of each sentence
        take_log: Whether to take log of KL divergences
        save_results: Whether to save results to disk
        output_dir: Directory to save results
        max_problems: Maximum number of problems to process (None for all)
        include_incorrect: Whether to include incorrect solutions

    Returns:
        Dictionary mapping (problem_num, is_correct) to KL matrices
    """
    # Get all problem numbers
    problem_list = get_all_problem_numbers(model_name, include_incorrect)

    if max_problems:
        problem_list = problem_list[:max_problems]

    print(f"Found {len(problem_list)} problems to process")

    results = {}

    for problem_num, is_correct in tqdm(problem_list, desc="Processing problems"):
        correct_str = "correct" if is_correct else "incorrect"
        print(f"\nProcessing problem {problem_num} ({correct_str})...")

        try:
            # Get sentence-to-sentence KL divergences
            sentence_sentence_scores = get_suppression_KL_matrix(
                problem_num=problem_num,
                p_nucleus=p_nucleus,
                model_name=model_name,
                is_correct=is_correct,
                only_first=only_first,
                take_log=take_log,
            )

            if sentence_sentence_scores is None:
                print(f"  Skipping {problem_num} ({correct_str}) - returned None")
                continue

            print(f"  Computed KL matrix shape: {sentence_sentence_scores.shape}")
            results[(problem_num, is_correct)] = sentence_sentence_scores

            # Save individual result if requested
            if save_results:
                save_path = Path(output_dir) / model_name / correct_str
                save_path.mkdir(parents=True, exist_ok=True)
                np.save(save_path / f"problem_{problem_num}_kl.npy", sentence_sentence_scores)
                print(f"  Saved to {save_path / f'problem_{problem_num}_kl.npy'}")

        except Exception as e:
            print(f"  Error processing problem {problem_num} ({correct_str}): {e}")
            continue

    # Save summary results
    if save_results and results:
        summary_path = Path(output_dir) / model_name / "summary.json"
        summary = {
            "model_name": model_name,
            "num_problems": len(results),
            "p_nucleus": p_nucleus,
            "take_log": take_log,
            "only_first": only_first,
            "problems": [
                {"problem_num": pn, "is_correct": ic, "shape": results[(pn, ic)].shape}
                for pn, ic in results.keys()
            ],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {summary_path}")

    return results



if __name__ == "__main__":
    # Testing
    model_name = "qwen-15b"
    p_nucleus = 0.9999
    take_log = True
    only_first = None  # Process all tokens in each sentence
    plot_sentences = False  # Set to True to generate plots for each sentence
    max_problems = None  # Set to a number to limit processing (e.g., 10 for testing)
    include_incorrect = True  # Process both correct and incorrect solutions

    # Process all problems
    results = process_all_problems_kl(
        model_name=model_name,
        p_nucleus=p_nucleus,
        only_first=only_first,
        take_log=take_log,
        save_results=True,
        output_dir="kl_results",
        max_problems=max_problems,
        include_incorrect=include_incorrect,
    )
