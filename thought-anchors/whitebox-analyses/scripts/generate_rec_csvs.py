"""
Generate CSV files with receiver head scores for each sentence in COT reasoning.

This script outputs CSV files containing:
- Problem number
- Correct/incorrect status
- Sentence text
- Receiver head scores
- Taxonomic labels for each sentence
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from scipy import stats

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from attention_analysis.receiver_head_funcs import (
    get_top_k_receiver_heads,
    get_problem_text_sentences,
    get_all_heads_vert_scores,
    get_receiver_head_scores,
    get_model_rollouts_root,
)
from attention_analysis.attn_funcs import get_vertical_scores


def get_taxonomic_labels(
    problem_num: int, is_correct: bool, model_name: str = "qwen-14b"
) -> List[List[str]]:
    """Get taxonomic labels for each sentence/chunk in the problem."""
    dir_root = get_model_rollouts_root(model_name)
    if is_correct:
        ci = "correct_base_solution"
    else:
        ci = "incorrect_base_solution"

    dir_problem = os.path.join(dir_root, ci, f"problem_{problem_num}")
    fp_chunks_labeled = os.path.join(dir_problem, "chunks_labeled.json")

    if not os.path.exists(fp_chunks_labeled):
        print(
            f"Warning: chunks_labeled.json not found for problem {problem_num} ({ci})"
        )
        return []

    with open(fp_chunks_labeled, "r") as f:
        chunks_labeled = json.load(f)

    # Extract function_tags for each chunk
    labels = [chunk.get("function_tags", []) for chunk in chunks_labeled]
    return labels


def get_all_problems_list(
    model_name: str = "qwen-14b",
) -> List[Tuple[int, bool]]:
    """Get list of all problems and their correctness status."""
    dir_root = get_model_rollouts_root(model_name)
    problems_list = []

    # Check correct solutions
    correct_dir = os.path.join(dir_root, "correct_base_solution")
    if os.path.exists(correct_dir):
        for problem_dir in os.listdir(correct_dir):
            if (
                "problem_3935" in problem_dir
            ):  # 13k tokens long, too intense on the RAM/VRAM
                continue

            if problem_dir.startswith("problem_"):
                problem_num = int(problem_dir.replace("problem_", ""))
                problems_list.append((problem_num, True))

    # Check incorrect solutions
    incorrect_dir = os.path.join(dir_root, "incorrect_base_solution")
    if os.path.exists(incorrect_dir):
        for problem_dir in os.listdir(incorrect_dir):
            if (
                "problem_3935" in problem_dir
            ):  # 13k tokens long, too intense on the RAM/VRAM
                continue
            if problem_dir.startswith("problem_"):
                problem_num = int(problem_dir.replace("problem_", ""))
                problems_list.append((problem_num, False))

    return sorted(problems_list)


def detect_convergence(
    problem_num: int,
    is_correct: bool,
    model_name: str = "qwen-14b",
    convergence_window: int = 5,
    high_threshold: float = 0.95,
    low_threshold: float = 0.05,
) -> List[bool]:
    """
    Detect convergence for each sentence in the problem.
    Convergence is defined as accuracy > high_threshold or < low_threshold
    for convergence_window consecutive sentences.

    Returns a list of booleans where True means the sentence is pre-convergence.
    """
    dir_root = get_model_rollouts_root(model_name)
    if is_correct:
        ci = "correct_base_solution"
    else:
        ci = "incorrect_base_solution"

    dir_problem = os.path.join(dir_root, ci, f"problem_{problem_num}")
    fp_chunks_labeled = os.path.join(dir_problem, "chunks_labeled.json")

    if not os.path.exists(fp_chunks_labeled):
        print(
            f"Warning: chunks_labeled.json not found for problem {problem_num} ({ci})"
        )
        return []

    with open(fp_chunks_labeled, "r") as f:
        chunks_labeled = json.load(f)

    # Extract accuracies
    accuracies = [chunk.get("accuracy", 0.5) for chunk in chunks_labeled]
    n_chunks = len(accuracies)

    # Find convergence point
    convergence_idx = n_chunks  # Default to no convergence

    for i in range(n_chunks - convergence_window + 1):
        window_accuracies = accuracies[i : i + convergence_window]

        # Check if all accuracies in window are above high threshold
        if all(acc > high_threshold for acc in window_accuracies):
            convergence_idx = i
            break

        # Check if all accuracies in window are below low threshold
        if all(acc < low_threshold for acc in window_accuracies):
            convergence_idx = i
            break

    # Create boolean list: True for pre-convergence, False for post-convergence
    # Note: we exclude the last chunk (final answer) so we have n_chunks - 1 sentences
    pre_convergence = [i < convergence_idx for i in range(n_chunks - 1)]

    return pre_convergence, accuracies


def calculate_receiver_head_scores_for_problem(
    problem_num: int,
    is_correct: bool,
    model_name: str = "qwen-15b",  # Using qwen-15b for analysis
    top_k: int = 16,
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> np.ndarray:
    """
    Calculate receiver head scores for a single problem.
    Returns an array of scores for each sentence.
    """
    # Get problem text and sentences
    text, sentences_w_spacing = get_problem_text_sentences(
        problem_num, is_correct, model_name
    )  # Load from qwen-14b data

    # Get top k receiver heads (these are the most important attention heads)
    print(f"  Getting top {top_k} receiver heads for {model_name}...")
    top_k_layer_head = get_top_k_receiver_heads(
        model_name=model_name,
        top_k=top_k,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
    )

    # Calculate vertical scores for all heads for this problem
    print(
        f"  Calculating vertical scores for problem {problem_num} (is_correct={is_correct})..."
    )
    layer_head_vert_scores = get_all_heads_vert_scores(
        text,
        sentences_w_spacing,
        model_name=model_name,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
        score_type="mean",
    )

    # Get receiver head scores using the top-k heads
    rec_head_scores = get_receiver_head_scores(
        top_k_layer_head, layer_head_vert_scores
    )

    return rec_head_scores


def generate_receiver_head_csvs(
    model_name: str = "qwen-15b",  # Model for analysis
    data_model: str = "qwen-14b",  # Model for data source
    top_k: int = 16,
    proximity_ignore: int = 4,
    control_depth: bool = False,
    output_dir: str = "csvs",  # Output to outer csvs folder
    max_problems: Optional[int] = None,
):
    """
    Generate CSV files with receiver head scores for all problems.
    """
    # Create output directory at the outermost level (relative to whitebox-analyses/scripts/)
    output_path = Path(__file__).parent.parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of all problems
    problems_list = get_all_problems_list(data_model)
    if max_problems:
        problems_list = problems_list[:max_problems]

    print(f"Processing {len(problems_list)} problems...")

    all_data = []

    for problem_num, is_correct in tqdm(
        problems_list, desc="Processing problems"
    ):
        try:
            # Get problem text and sentences
            text, sentences_w_spacing = get_problem_text_sentences(
                problem_num, is_correct, data_model
            )

            # Get taxonomic labels
            labels = get_taxonomic_labels(problem_num, is_correct, data_model)

            # Detect convergence for each sentence
            pre_convergence, accuracies = detect_convergence(
                problem_num, is_correct, data_model
            )

            # Calculate receiver head scores
            rec_head_scores = calculate_receiver_head_scores_for_problem(
                problem_num,
                is_correct,
                model_name=model_name,
                top_k=top_k,
                proximity_ignore=proximity_ignore,
                control_depth=control_depth,
            )
            labels = labels[:-1]

            # Ensure we have the same number of sentences, labels, and scores
            n_sentences = len(sentences_w_spacing)
            if len(labels) != n_sentences:
                print(
                    f"Warning: Number of labels ({len(labels)}) doesn't match sentences ({n_sentences}) for problem {problem_num}"
                )
                # Pad or truncate labels to match
                if len(labels) < n_sentences:
                    labels.extend([[]] * (n_sentences - len(labels)))
                else:
                    labels = labels[:n_sentences]

            if len(rec_head_scores) != n_sentences:
                print(
                    f"Warning: Number of receiver scores ({len(rec_head_scores)}) doesn't match sentences ({n_sentences}) for problem {problem_num}"
                )
                # This shouldn't happen, but handle it just in case
                continue

            # Check and align pre_convergence list
            if len(pre_convergence) != n_sentences:
                print(
                    f"Warning: Number of pre_convergence values ({len(pre_convergence)}) doesn't match sentences ({n_sentences}) for problem {problem_num}"
                )
                # Pad with True (assume pre-convergence) if needed
                if len(pre_convergence) < n_sentences:
                    pre_convergence.extend(
                        [True] * (n_sentences - len(pre_convergence))
                    )
                else:
                    pre_convergence = pre_convergence[:n_sentences]

            # Create rows for each sentence
            prior_acc = None
            for idx, (
                sentence,
                label_list,
                score,
                is_pre_conv,
                accuracy,
            ) in enumerate(
                zip(
                    sentences_w_spacing,
                    labels,
                    rec_head_scores,
                    pre_convergence,
                    accuracies,
                )
            ):
                if prior_acc is not None:
                    simple_importance = abs(accuracy - prior_acc)
                else:
                    simple_importance = None
                prior_acc = accuracy
                row = {
                    "problem_number": problem_num,
                    "is_correct": is_correct,
                    "sentence_idx": idx,
                    "sentence": sentence.strip(),
                    "receiver_head_score": float(
                        score
                    ),  # Convert numpy float to Python float
                    "taxonomic_labels": (
                        "|".join(label_list) if label_list else "none"
                    ),  # Join labels with |
                    "pre_convergence": is_pre_conv,  # Add convergence status
                    "accuracy": accuracy,  # Add accuracy
                    "simple_importance": simple_importance,  # Add simple importance
                }

                all_data.append(row)

        except Exception as e:
            print(
                f"Error processing problem {problem_num} (is_correct={is_correct}): {e}"
            )
            continue

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Save to CSV files
    # Save all data
    all_data_path = (
        output_path
        / f"receiver_head_scores_all_{model_name}_k{top_k}_pi{proximity_ignore}.csv"
    )
    df.to_csv(all_data_path, index=False)
    print(f"Saved all data to {all_data_path}")

    # Save correct solutions only
    df_correct = df[df["is_correct"] == True]
    correct_path = (
        output_path
        / f"receiver_head_scores_correct_{model_name}_k{top_k}_pi{proximity_ignore}.csv"
    )
    df_correct.to_csv(correct_path, index=False)
    print(f"Saved correct solutions to {correct_path}")

    # Save incorrect solutions only
    df_incorrect = df[df["is_correct"] == False]
    incorrect_path = (
        output_path
        / f"receiver_head_scores_incorrect_{model_name}_k{top_k}_pi{proximity_ignore}.csv"
    )
    df_incorrect.to_csv(incorrect_path, index=False)
    print(f"Saved incorrect solutions to {incorrect_path}")

    # Create summary statistics
    summary_stats = {
        "total_problems": len(problems_list),
        "total_sentences": len(df),
        "correct_problems": int(df_correct["problem_number"].nunique()),
        "incorrect_problems": int(df_incorrect["problem_number"].nunique()),
        "avg_sentences_per_problem": (
            float(len(df) / len(problems_list)) if problems_list else 0
        ),
        "avg_receiver_score": float(df["receiver_head_score"].mean()),
        "std_receiver_score": float(df["receiver_head_score"].std()),
    }

    # Save summary statistics
    summary_path = (
        output_path
        / f"receiver_head_summary_{model_name}_k{top_k}_pi{proximity_ignore}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Saved summary statistics to {summary_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CSV files with receiver head scores"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen-14b",
        help="Model name for analysis",
    )
    parser.add_argument(
        "--data-model",
        type=str,
        default="qwen-14b",
        help="Model name for data source",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Number of top receiver heads to use",
    )
    parser.add_argument(
        "--proximity-ignore",
        type=int,
        default=16,
        help="Proximity ignore for vertical scores",
    )
    parser.add_argument(
        "--control-depth",
        action="store_true",
        help="Control for depth in vertical scores",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="csvs",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Maximum number of problems to process (default: all)",
    )

    args = parser.parse_args()

    # Generate CSV files
    df = generate_receiver_head_csvs(
        model_name=args.model_name,
        data_model=args.data_model,
        top_k=args.top_k,
        proximity_ignore=args.proximity_ignore,
        control_depth=args.control_depth,
        output_dir=args.output_dir,
        max_problems=args.max_problems,
    )

    print(
        f"\nProcessing complete! Generated CSV files with {len(df)} total rows."
    )
