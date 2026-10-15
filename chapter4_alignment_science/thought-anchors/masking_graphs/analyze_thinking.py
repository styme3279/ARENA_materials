#!/usr/bin/env python
"""
Analyze MMLU thinking mode results and compare with non-thinking baseline.

This script:
1. Loads non-thinking results from CSV
2. Loads thinking results from JSON files
3. Combines them into a comprehensive dataframe
4. Performs various analyses
5. Saves the combined results
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from base_responses_thinking import get_parameter_specific_output_dir

from visualization_utils import (
    create_interactive_subject_scatter,
    create_interactive_difficulty_plot,
    create_confidence_calibration_plot,
)

INTERACTIVE_AVAILABLE = True
# Default configuration - can be overridden
DEFAULT_MAX_QUESTIONS = 16000
DEFAULT_MAX_NONTHINKING_ACCURACY = None  # None means all questions


def load_thinking_results(output_dir: str) -> Dict:
    """Load thinking results from batch files, keyed by hash_key."""
    thinking_results = {}

    # Load from batch files
    batch_files = sorted(list(Path(output_dir).glob("batch_*.json")))

    if batch_files:
        print(f"Loading thinking results from {len(batch_files)} batch files")

        for batch_file in tqdm(batch_files, desc="Loading batches"):
            with open(batch_file, "r") as f:
                batch_data = json.load(f)

            # Each batch is a list of results
            if isinstance(batch_data, list):
                for item in batch_data:
                    # Use hash_key as the unique identifier
                    if (
                        "question_data" in item
                        and "hash_key" in item["question_data"]
                    ):
                        hash_key = item["question_data"]["hash_key"]
                        thinking_results[hash_key] = item
                    else:
                        print(f"Warning: Missing hash_key in thinking result")
    else:
        print(f"No batch files found in {output_dir}")

    print(f"Loaded {len(thinking_results)} thinking results")
    return thinking_results


def load_nonthinking_results(
    model_base, csv_path: str = None, max_questions: int = None
) -> pd.DataFrame:
    """Load non-thinking baseline results from CSV."""
    if csv_path is None:
        model_str = model_base.split("/")[-1]
        output_dir = os.path.join("results", "mmlu_results", model_str)
        csv_path = os.path.join(output_dir, f"results_{max_questions}.csv")

    print(f"Loading non-thinking results from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} non-thinking results")
    return df


def extract_thinking_stats(thinking_data: Dict) -> Dict:
    """Extract statistics from a single thinking result."""
    stats = {
        "thinking_is_correct": thinking_data.get("is_correct", False),
        "thinking_majority_answer": thinking_data.get("majority_answer", ""),
        "thinking_confidence": thinking_data.get("confidence", 0.0),
        "thinking_answer_distribution": thinking_data.get(
            "answer_distribution", {}
        ),
        "thinking_all_is_correct": thinking_data.get("all_is_correct", []),
        "thinking_num_runs": thinking_data.get("successful_runs", 0),
        "thinking_accuracy": (
            np.mean(thinking_data.get("all_is_correct", [0]))
            if thinking_data.get("all_is_correct")
            else 0.0
        ),
        "thinking_consensus_strength": 0.0,
        "thinking_extraction_failures": thinking_data.get(
            "extraction_failures", 0
        ),
        "thinking_errors": len(thinking_data.get("errors", [])),
    }

    # Calculate consensus strength from answer_distribution
    answer_dist = thinking_data.get("answer_distribution", {})
    if answer_dist and thinking_data.get("majority_answer"):
        majority_count = answer_dist.get(thinking_data["majority_answer"], 0)
        total_votes = sum(answer_dist.values())
        if total_votes > 0:
            stats["thinking_consensus_strength"] = majority_count / total_votes

    # Extract thinking trace statistics if available
    if "all_thinking_traces" in thinking_data:
        traces = thinking_data["all_thinking_traces"]
        if traces:
            # Calculate average thinking trace length
            trace_lengths = [len(trace) if trace else 0 for trace in traces]
            stats["thinking_avg_trace_length"] = (
                np.mean(trace_lengths) if trace_lengths else 0
            )
            stats["thinking_min_trace_length"] = (
                min(trace_lengths) if trace_lengths else 0
            )
            stats["thinking_max_trace_length"] = (
                max(trace_lengths) if trace_lengths else 0
            )
    elif "avg_thinking_length" in thinking_data:
        stats["thinking_avg_trace_length"] = thinking_data[
            "avg_thinking_length"
        ]

    # Extract token usage statistics if available
    if "total_usage" in thinking_data:
        usage = thinking_data["total_usage"]
        if usage and isinstance(usage, dict):
            stats["thinking_total_tokens"] = usage.get("total_tokens", 0)
            stats["thinking_prompt_tokens"] = usage.get("prompt_tokens", 0)
            stats["thinking_completion_tokens"] = usage.get(
                "completion_tokens", 0
            )

    return stats


def combine_results(
    nonthinking_df: pd.DataFrame, thinking_results: Dict
) -> pd.DataFrame:
    """Combine non-thinking and thinking results using hash_key for proper pairing."""

    # Start with non-thinking results as base
    combined_df = nonthinking_df.copy()

    # Add thinking columns with default values
    thinking_columns = [
        "thinking_is_correct",
        "thinking_majority_answer",
        "thinking_accuracy",
        "thinking_consensus_strength",
        "thinking_num_runs",
        "thinking_confidence",
        "thinking_extraction_failures",
        "thinking_errors",
        "thinking_avg_trace_length",
        "thinking_min_trace_length",
        "thinking_max_trace_length",
        "thinking_total_tokens",
        "thinking_prompt_tokens",
        "thinking_completion_tokens",
    ]

    for col in thinking_columns:
        combined_df[col] = None

    # Match thinking results using hash_key
    matched_count = 0
    unmatched_thinking = []

    for hash_key, thinking_data in thinking_results.items():
        # Find the corresponding row in nonthinking_df by hash_key
        matching_rows = combined_df[combined_df["hash_key"] == hash_key]

        if len(matching_rows) == 1:
            idx = matching_rows.index[0]
            thinking_stats = extract_thinking_stats(thinking_data)

            for key, value in thinking_stats.items():
                # Skip complex data types (lists, dicts) that can't be stored in dataframe cells
                if key not in [
                    "thinking_answer_distribution",
                    "thinking_all_is_correct",
                ] and not isinstance(value, (dict, list)):
                    if (
                        key in thinking_columns
                    ):  # Only add columns we've defined
                        combined_df.at[idx, key] = value
            matched_count += 1
        elif len(matching_rows) == 0:
            unmatched_thinking.append(hash_key)
        else:
            print(f"Warning: Multiple matches for hash_key {hash_key}")

    print(f"Matched {matched_count} thinking results to non-thinking results")
    if unmatched_thinking:
        print(
            f"Warning: {len(unmatched_thinking)} thinking results had no match in non-thinking data"
        )
        if len(unmatched_thinking) <= 10:
            print(f"  Unmatched hash_keys: {unmatched_thinking}")

    # Calculate improvement metrics
    # thinking_improvement is now the difference between thinking accuracy and prob_correct
    combined_df["thinking_improvement"] = combined_df[
        "thinking_is_correct"
    ].astype(float) - combined_df["prob_correct"].astype(float)

    combined_df["has_thinking_result"] = ~combined_df[
        "thinking_is_correct"
    ].isna()

    return combined_df


def analyze_results(df: pd.DataFrame) -> Dict:
    """Perform comprehensive analysis of combined results."""

    analysis = {}

    # Overall statistics
    analysis["total_questions"] = len(df)
    analysis["questions_with_thinking"] = df["has_thinking_result"].sum()

    # PAIRED COMPARISON - Only questions with both thinking and non-thinking results
    paired_df = df[df["has_thinking_result"]].copy()

    if len(paired_df) > 0:
        # Paired comparison statistics
        analysis["paired_comparison"] = {
            "num_questions": len(paired_df),
            "nonthinking_accuracy": paired_df["is_correct"].mean(),
            "thinking_accuracy": paired_df["thinking_is_correct"].mean(),
            "absolute_improvement": paired_df["thinking_is_correct"].mean()
            - paired_df["prob_correct"].mean(),
            "avg_prob_correct": paired_df["prob_correct"].mean(),
            "avg_prob_generated": (
                paired_df["prob_generated"].mean()
                if "prob_generated" in paired_df.columns
                else None
            ),
            "improvement_rate": (paired_df["thinking_improvement"] > 0).mean(),
            "degradation_rate": (paired_df["thinking_improvement"] < 0).mean(),
            "no_change_rate": (paired_df["thinking_improvement"] == 0).mean(),
            "avg_consensus": paired_df["thinking_consensus_strength"].mean(),
        }

        # Calculate relative improvement percentage
        if analysis["paired_comparison"]["nonthinking_accuracy"] > 0:
            analysis["paired_comparison"]["relative_improvement_pct"] = (
                analysis["paired_comparison"]["absolute_improvement"]
                / analysis["paired_comparison"]["nonthinking_accuracy"]
                * 100
            )
        else:
            # If baseline is 0, any correct answer is infinite improvement
            analysis["paired_comparison"]["relative_improvement_pct"] = (
                float("inf")
                if analysis["paired_comparison"]["thinking_accuracy"] > 0
                else 0
            )

        # Token usage for paired questions
        if "thinking_total_tokens" in paired_df.columns:
            analysis["paired_comparison"]["avg_tokens_per_question"] = (
                paired_df["thinking_total_tokens"].mean()
            )
            analysis["paired_comparison"]["total_tokens_used"] = paired_df[
                "thinking_total_tokens"
            ].sum()

    # Overall non-thinking performance (all questions, for reference)
    analysis["overall_nonthinking_accuracy"] = df["is_correct"].mean()
    analysis["overall_nonthinking_prob_correct_mean"] = df[
        "prob_correct"
    ].mean()

    # Performance by difficulty (using prob_generated as proxy) - PAIRED ONLY
    # Note: prob_generated represents the model's confidence in its chosen answer
    # prob_correct is the probability assigned to the correct answer (different!)
    # Low confidence = hard question (model uncertain)
    # High confidence = easy question (model confident)
    difficulty_bins = [0, 0.25, 0.5, 0.75, 1.0]

    # Use prob_generated if available, otherwise fall back to prob_correct
    if "prob_generated" in paired_df.columns:
        confidence_col = "prob_generated"
    else:
        print(
            "Warning: prob_generated not found, using prob_correct as fallback"
        )
        confidence_col = "prob_correct"

    paired_df["difficulty_bin"] = pd.cut(
        paired_df[confidence_col],
        bins=difficulty_bins,
        labels=["Hard", "Medium-Hard", "Medium-Easy", "Easy"],
        include_lowest=True,
    )

    analysis["paired_performance_by_difficulty"] = {}
    for difficulty in ["Hard", "Medium-Hard", "Medium-Easy", "Easy"]:
        subset = paired_df[paired_df["difficulty_bin"] == difficulty]

        if len(subset) > 0:
            analysis["paired_performance_by_difficulty"][difficulty] = {
                "count": len(subset),
                "nonthinking_accuracy": subset["is_correct"].mean(),
                "thinking_accuracy": subset["thinking_is_correct"].mean(),
                "absolute_improvement": subset["thinking_is_correct"].mean()
                - subset["prob_correct"].mean(),
                "improvement_rate": (subset["thinking_improvement"] > 0).mean(),
                "degradation_rate": (subset["thinking_improvement"] < 0).mean(),
                "no_change_rate": (subset["thinking_improvement"] == 0).mean(),
            }

    # Performance by subject - PAIRED ONLY
    if "subject" in paired_df.columns and len(paired_df) > 0:
        analysis["paired_performance_by_subject"] = {}
        for subject in paired_df["subject"].unique():
            subset = paired_df[paired_df["subject"] == subject]

            if len(subset) > 0:
                analysis["paired_performance_by_subject"][subject] = {
                    "count": len(subset),
                    "nonthinking_accuracy": subset["is_correct"].mean(),
                    "thinking_accuracy": subset["thinking_is_correct"].mean(),
                    "absolute_improvement": subset["thinking_is_correct"].mean()
                    - subset["prob_correct"].mean(),
                    "improvement_rate": (
                        subset["thinking_improvement"] > 0
                    ).mean(),
                    "degradation_rate": (
                        subset["thinking_improvement"] < 0
                    ).mean(),
                    "no_change_rate": (
                        subset["thinking_improvement"] == 0
                    ).mean(),
                }

    return analysis


def create_visualizations(
    df: pd.DataFrame,
    model_base,
    model_thinking,
    output_dir: str = "results/thinking_vs_nonthinking",
):
    """Create visualization plots for the analysis."""

    os.makedirs(output_dir, exist_ok=True)

    # Only analyze questions with thinking results
    thinking_df = df[df["has_thinking_result"]].copy()

    if len(thinking_df) == 0:
        print("No thinking results to visualize")
        return

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. Paired Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    accuracies = pd.DataFrame(
        {
            "Non-thinking": [thinking_df["is_correct"].mean()],
            "Thinking": [thinking_df["thinking_is_correct"].mean()],
        }
    )
    accuracies.plot(kind="bar", ax=ax, color=["#FF6B6B", "#4ECDC4"])
    ax.set_title(
        f"Paired Comparison: Non-thinking vs Thinking (n={len(thinking_df)})\n{model_base} vs {model_thinking}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_xticklabels([""], rotation=0)
    ax.legend(title="Mode")
    ax.set_ylim(0, max(1, accuracies.max().max() * 1.1))

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.png", dpi=300)
    plt.close()

    # 2. Improvement distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    improvements = thinking_df["thinking_improvement"].dropna()
    ax.hist(
        improvements,
        bins=[-1, -0.5, 0, 0.5, 1],
        color="#95E77E",
        edgecolor="black",
        alpha=0.7,
    )
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No change")
    ax.set_title(
        f"Distribution of Thinking Mode Improvements\n{model_base} → {model_thinking}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Improvement (Thinking Accuracy - Model Confidence)")
    ax.set_ylabel("Number of Questions")
    ax.legend()

    # Add statistics text
    improve_rate = (improvements > 0).mean() * 100
    degrade_rate = (improvements < 0).mean() * 100
    no_change_rate = (improvements == 0).mean() * 100

    stats_text = f"Improved: {improve_rate:.1f}%\nDegraded: {degrade_rate:.1f}%\nNo Change: {no_change_rate:.1f}%"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement_distribution.png", dpi=300)
    plt.close()

    # 3. Performance by difficulty
    fig, ax = plt.subplots(figsize=(12, 6))

    if "difficulty_bin" in thinking_df.columns:
        # Get all possible difficulty categories
        all_difficulties = [
            "Hard (<25%)",
            "Medium-Hard (25-50%)",
            "Medium-Easy (50-75%)",
            "Easy (75-100%)",
        ]

        # Check what's actually in the data
        actual_difficulties = thinking_df[
            "difficulty_bin"
        ].cat.categories.tolist()

        difficulty_data = []
        for difficulty in actual_difficulties:
            subset = thinking_df[thinking_df["difficulty_bin"] == difficulty]
            if len(subset) > 0:
                # Map old labels to new labels if needed
                label_map = {
                    "Hard": "Hard (<25%)",
                    "Medium-Hard": "Medium-Hard (25-50%)",
                    "Medium-Easy": "Medium-Easy (50-75%)",
                    "Easy": "Easy (75-100%)",
                }
                display_label = label_map.get(difficulty, difficulty)

                difficulty_data.append(
                    {
                        "Difficulty": display_label,
                        "Non-thinking": subset["is_correct"].mean(),
                        "Thinking": subset["thinking_is_correct"].mean(),
                        "Count": len(subset),
                    }
                )

        if difficulty_data:
            diff_df = pd.DataFrame(difficulty_data)

            # Create grouped bar chart
            x = np.arange(len(diff_df))
            width = 0.35

            bars1 = ax.bar(
                x - width / 2,
                diff_df["Non-thinking"],
                width,
                label="Non-thinking",
                color="#FF6B6B",
            )
            bars2 = ax.bar(
                x + width / 2,
                diff_df["Thinking"],
                width,
                label="Thinking",
                color="#4ECDC4",
            )

            ax.set_title(
                f"Accuracy by Question Difficulty (Model Confidence)\n{model_base} vs {model_thinking}",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Difficulty Level")
            ax.set_xticks(x)
            ax.set_xticklabels(diff_df["Difficulty"], rotation=45, ha="right")
            ax.legend(title="Mode")
            ax.set_ylim(0, 1)

            # Add value labels on bars
            ax.bar_label(bars1, fmt="%.3f")
            ax.bar_label(bars2, fmt="%.3f")

            # Add count labels below
            for i, count in enumerate(diff_df["Count"]):
                ax.text(
                    i,
                    -0.05,
                    f"n={count}",
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No difficulty data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Difficulty binning not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_by_difficulty.png", dpi=300)
    plt.close()

    # 4. Consensus strength distribution
    if "thinking_consensus_strength" in thinking_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        consensus = thinking_df["thinking_consensus_strength"].dropna()
        ax.hist(
            consensus, bins=20, color="#FFE66D", edgecolor="black", alpha=0.7
        )
        ax.axvline(
            x=consensus.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {consensus.mean():.3f}",
        )
        ax.set_title(
            f"Distribution of Consensus Strength in Thinking Mode\n{model_thinking}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(
            "Consensus Strength (Fraction voting for majority answer)"
        )
        ax.set_ylabel("Number of Questions")
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/consensus_distribution.png", dpi=300)
        plt.close()

    # 5. Performance by subject (if available)
    if "subject" in thinking_df.columns:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Get top 10 subjects by question count
        top_subjects = thinking_df["subject"].value_counts().head(10).index

        subject_data = []
        for subject in top_subjects:
            subset = thinking_df[thinking_df["subject"] == subject]
            subject_data.append(
                {
                    "Subject": subject[:30],  # Truncate long names
                    "Non-thinking": subset["is_correct"].mean(),
                    "Thinking": subset["thinking_is_correct"].mean(),
                    "Count": len(subset),
                }
            )

        subj_df = pd.DataFrame(subject_data)
        subj_df.set_index("Subject")[["Non-thinking", "Thinking"]].plot(
            kind="bar", ax=ax, color=["#FF6B6B", "#4ECDC4"]
        )
        ax.set_title(
            f"Accuracy by Subject (Top 10)\n{model_base} vs {model_thinking}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Subject")
        ax.legend(title="Mode")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")

        # Add count annotations
        for i, (idx, row) in enumerate(subj_df.iterrows()):
            ax.text(
                i,
                -0.15,
                f'n={row["Count"]}',
                ha="center",
                transform=ax.get_xaxis_transform(),
            )

        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_by_subject.png", dpi=300)
        plt.close()

    # 6. Scatter plot of all subjects - Thinking vs Non-thinking accuracy
    if "subject" in thinking_df.columns:
        fig, ax = plt.subplots(figsize=(16, 16))  # Perfect square

        # Calculate accuracy for each subject
        subject_accuracies = []
        for subject in thinking_df["subject"].unique():
            subset = thinking_df[thinking_df["subject"] == subject]
            if (
                len(subset) >= 10
            ):  # Only include subjects with at least 10 questions
                nonthinking_acc = subset["is_correct"].mean()
                thinking_acc = subset["thinking_is_correct"].mean()
                subject_accuracies.append(
                    {
                        "subject": subject,
                        "nonthinking": nonthinking_acc,
                        "thinking": thinking_acc,
                        "count": len(subset),
                        "improvement": thinking_acc - nonthinking_acc,
                    }
                )

        # Sort by improvement for consistent coloring
        subject_accuracies.sort(key=lambda x: x["improvement"], reverse=True)

        # Create scatter plot
        for data in subject_accuracies:
            # Color based on improvement
            if data["improvement"] > 0.1:
                color = "#2ecc71"  # Green for large improvement
                marker = "^"
            elif data["improvement"] > 0:
                color = "#3498db"  # Blue for small improvement
                marker = "o"
            elif data["improvement"] > -0.05:
                color = "#f39c12"  # Orange for small degradation
                marker = "v"
            else:
                color = "#e74c3c"  # Red for large degradation
                marker = "v"

            # Size based on number of questions
            size = min(500, 50 + data["count"] * 0.5)

            ax.scatter(
                data["nonthinking"],
                data["thinking"],
                s=size,
                c=color,
                alpha=0.7,
                marker=marker,
                edgecolors="black",
                linewidth=0.5,
            )

            # Add text label with subject name
            # Adjust position to avoid overlap
            offset_x = 0.01
            offset_y = 0.01
            if data["improvement"] < 0:
                offset_y = -0.015

            ax.text(
                data["nonthinking"] + offset_x,
                data["thinking"] + offset_y,
                data["subject"].replace("_", " ").title()[:25],
                fontsize=9,
                alpha=0.9,
                fontweight="bold",
            )

        # Find the minimum accuracy for better bounds
        all_accuracies = [d["nonthinking"] for d in subject_accuracies] + [
            d["thinking"] for d in subject_accuracies
        ]
        min_accuracy = min(all_accuracies) - 0.05
        max_accuracy = max(all_accuracies) + 0.05

        # Add diagonal line (y=x) for reference
        ax.plot(
            [min_accuracy, max_accuracy],
            [min_accuracy, max_accuracy],
            "k--",
            alpha=0.3,
            linewidth=1,
        )

        # Add improvement zones
        ax.fill_between(
            [min_accuracy, max_accuracy],
            [min_accuracy, max_accuracy],
            [max_accuracy, max_accuracy],
            alpha=0.05,
            color="green",
            label="Improvement Zone",
        )
        ax.fill_between(
            [min_accuracy, max_accuracy],
            [min_accuracy, min_accuracy],
            [min_accuracy, max_accuracy],
            alpha=0.05,
            color="red",
            label="Degradation Zone",
        )

        ax.set_xlabel("Non-thinking Accuracy", fontsize=18, fontweight="bold")
        ax.set_ylabel("Thinking Accuracy", fontsize=18, fontweight="bold")
        ax.set_title(
            f"Subject Performance: Thinking vs Non-thinking Mode\n{model_base} → {model_thinking}",
            fontsize=20,
            fontweight="bold",
        )

        # Set equal aspect ratio and limits based on data range
        ax.set_aspect("equal")
        ax.set_xlim(min_accuracy, max_accuracy)
        ax.set_ylim(min_accuracy, max_accuracy)

        # Increase tick label font sizes
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.tick_params(axis="both", which="minor", labelsize=12)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add legend for marker meanings
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="#2ecc71",
                markersize=10,
                label="Large Improvement (>10%)",
                markeredgecolor="black",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#3498db",
                markersize=10,
                label="Small Improvement (0-10%)",
                markeredgecolor="black",
            ),
            Line2D(
                [0],
                [0],
                marker="v",
                color="w",
                markerfacecolor="#f39c12",
                markersize=10,
                label="Small Degradation (0-5%)",
                markeredgecolor="black",
            ),
            Line2D(
                [0],
                [0],
                marker="v",
                color="w",
                markerfacecolor="#e74c3c",
                markersize=10,
                label="Large Degradation (>5%)",
                markeredgecolor="black",
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

        # Add statistics text
        improvements = [d["improvement"] for d in subject_accuracies]
        avg_improvement = np.mean(improvements) * 100
        num_improved = sum(1 for i in improvements if i > 0)
        num_degraded = sum(1 for i in improvements if i < 0)

        stats_text = (
            f"Subjects analyzed: {len(subject_accuracies)}\n"
            f"Average improvement: {avg_improvement:.1f}%\n"
            f"Subjects improved: {num_improved}\n"
            f"Subjects degraded: {num_degraded}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(f"{output_dir}/subject_scatter_plot.png", dpi=200)
        plt.close()

    # 6b. Create confidence calibration plot (static)
    if "prob_generated" in thinking_df.columns:
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create confidence bins
        n_bins = 20
        thinking_df_copy = thinking_df.copy()
        thinking_df_copy["confidence_bin"] = pd.cut(
            thinking_df_copy["prob_generated"],
            bins=n_bins,
            labels=False,
            include_lowest=True,
        )

        # Calculate statistics for each bin
        bin_centers = []
        actual_accuracies = []
        bin_counts = []

        for bin_idx in range(n_bins):
            subset = thinking_df_copy[
                thinking_df_copy["confidence_bin"] == bin_idx
            ]
            if len(subset) > 0:
                bin_centers.append(subset["prob_generated"].mean())
                actual_accuracies.append(subset["is_correct"].mean())
                bin_counts.append(len(subset))

        # Create scatter plot
        if bin_centers:
            # Size based on count
            sizes = [min(500, 20 + c * 0.05) for c in bin_counts]

            # Color based on calibration error
            calibration_errors = [
                conf - acc for conf, acc in zip(bin_centers, actual_accuracies)
            ]
            colors = []
            for error in calibration_errors:
                if abs(error) < 0.05:
                    colors.append("#2ecc71")  # Well calibrated
                elif abs(error) < 0.1:
                    colors.append("#f39c12")  # Slightly miscalibrated
                elif error > 0:
                    colors.append("#e74c3c")  # Overconfident
                else:
                    colors.append("#3498db")  # Underconfident

            ax.scatter(
                bin_centers,
                actual_accuracies,
                s=sizes,
                c=colors,
                alpha=0.7,
                edgecolors="black",
            )

            # Labels and formatting
            ax.set_xlabel("Model Confidence (prob_generated)", fontsize=14)
            ax.set_ylabel("Actual Accuracy", fontsize=14)
            ax.set_title(
                f"Model Confidence Calibration\n{model_base} (Non-thinking Mode)",
                fontsize=16,
            )

            # Set limits based on data
            ax.set_xlim(min(bin_centers) - 0.05, max(bin_centers) + 0.05)
            ax.set_ylim(
                min(actual_accuracies) - 0.05, max(actual_accuracies) + 0.05
            )
            ax.grid(True, alpha=0.3)

            # Increase tick font sizes
            ax.tick_params(axis="both", which="major", labelsize=12)

            # Calculate ECE (Expected Calibration Error)
            ece = sum(
                abs(conf - acc) * count
                for conf, acc, count in zip(
                    bin_centers, actual_accuracies, bin_counts
                )
            ) / sum(bin_counts)
            mean_confidence = sum(
                conf * count for conf, count in zip(bin_centers, bin_counts)
            ) / sum(bin_counts)
            mean_accuracy = sum(
                acc * count for acc, count in zip(actual_accuracies, bin_counts)
            ) / sum(bin_counts)

            # Add statistics box
            stats_text = (
                f"ECE: {ece:.3f}\n"
                f"Avg Confidence: {mean_confidence:.1%}\n"
                f"Avg Accuracy: {mean_accuracy:.1%}\n"
                f"Total Questions: {sum(bin_counts)}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # Add legend for colors
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="#2ecc71", label="Well calibrated (±5%)"),
                Patch(facecolor="#f39c12", label="Slightly off (±5-10%)"),
                Patch(facecolor="#e74c3c", label="Overconfident (>10%)"),
                Patch(facecolor="#3498db", label="Underconfident (<-10%)"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/confidence_calibration.png", dpi=200)
            plt.close()

    # 7. Create interactive HTML visualizations if available
    if INTERACTIVE_AVAILABLE:
        try:
            # Create interactive subject scatter plot
            create_interactive_subject_scatter(
                thinking_df, f"{output_dir}/subject_scatter_interactive.html"
            )

            # Create interactive difficulty plot
            create_interactive_difficulty_plot(
                thinking_df,
                output_path=f"{output_dir}/difficulty_interactive.html",
            )

            # Create confidence calibration plots (both bins and subjects modes)
            create_confidence_calibration_plot(
                thinking_df,
                output_path=f"{output_dir}/confidence_calibration_bins.html",
                mode="bins",
            )
            create_confidence_calibration_plot(
                thinking_df,
                output_path=f"{output_dir}/confidence_calibration_subjects.html",
                mode="subjects",
            )
        except Exception as e:
            print(f"Warning: Could not create interactive plots: {e}")

    print(f"Visualizations saved to {output_dir}/")


def print_analysis_report(analysis: Dict):
    """Print a formatted analysis report."""

    print("\n" + "=" * 60)
    print("MMLU THINKING MODE ANALYSIS REPORT")
    print("=" * 60)

    print(f"\n📊 DATASET OVERVIEW")
    print(f"  Total questions: {analysis['total_questions']}")
    print(
        f"  Questions with thinking results: {analysis['questions_with_thinking']}"
    )
    print(
        f"  Coverage: {analysis['questions_with_thinking']/analysis['total_questions']*100:.1f}%"
    )

    # PAIRED COMPARISON - The main results
    if "paired_comparison" in analysis:
        pc = analysis["paired_comparison"]
        print(f"\n🎯 PAIRED COMPARISON RESULTS (n={pc['num_questions']})")
        print(
            f"  Note: Improvement is measured as (Thinking Accuracy - Model Confidence)"
        )
        print(f"  Average baseline prob_correct: {pc['avg_prob_correct']:.3f}")
        if "avg_prob_generated" in pc:
            print(
                f"  Average baseline confidence (prob_generated): {pc['avg_prob_generated']:.3f}"
            )
        print(f"\n  Non-thinking accuracy: {pc['nonthinking_accuracy']:.3f}")
        print(f"  Thinking accuracy: {pc['thinking_accuracy']:.3f}")
        print(f"  Absolute improvement: {pc['absolute_improvement']:+.3f}")
        if pc["relative_improvement_pct"] == float("inf"):
            print(f"  Relative improvement: ∞ (from 0% baseline)")
        else:
            print(
                f"  Relative improvement: {pc['relative_improvement_pct']:+.1f}%"
            )

        print(f"\n  📈 Question-level changes:")
        print(f"    Improved: {pc['improvement_rate']*100:.1f}%")
        print(f"    Degraded: {pc['degradation_rate']*100:.1f}%")
        print(f"    No change: {pc['no_change_rate']*100:.1f}%")

        print(f"\n  🤝 Consensus strength: {pc['avg_consensus']:.3f}")

        if "avg_tokens_per_question" in pc:
            print(f"\n  💭 Token usage:")
            print(
                f"    Average per question: {pc['avg_tokens_per_question']:.0f}"
            )
            print(f"    Total used: {pc['total_tokens_used']:,.0f}")

    if "paired_performance_by_difficulty" in analysis:
        print(f"\n📉 PAIRED PERFORMANCE BY DIFFICULTY")
        for difficulty, stats in analysis[
            "paired_performance_by_difficulty"
        ].items():
            print(f"\n  {difficulty} (n={stats['count']})")
            print(f"    Non-thinking: {stats['nonthinking_accuracy']:.3f}")
            print(f"    Thinking: {stats['thinking_accuracy']:.3f}")
            print(f"    Improvement: {stats['absolute_improvement']:+.3f}")
            print(
                f"    Improved/Degraded/Same: {stats['improvement_rate']*100:.0f}%/{stats['degradation_rate']*100:.0f}%/{stats['no_change_rate']*100:.0f}%"
            )

    if "paired_performance_by_subject" in analysis:
        print(f"\n📚 TOP 5 SUBJECTS BY IMPROVEMENT (PAIRED)")
        subject_improvements = [
            (subj, stats["absolute_improvement"], stats["count"])
            for subj, stats in analysis["paired_performance_by_subject"].items()
        ]
        subject_improvements.sort(key=lambda x: x[1], reverse=True)

        for subj, improvement, count in subject_improvements[:5]:
            print(f"  {subj[:40]:40} {improvement:+.3f} (n={count})")

        print(f"\n📚 BOTTOM 5 SUBJECTS BY IMPROVEMENT (PAIRED)")
        for subj, improvement, count in subject_improvements[-5:]:
            print(f"  {subj[:40]:40} {improvement:+.3f} (n={count})")

    print("\n" + "=" * 60)


def get_combined_df(
    model_base,
    model_thinking,
    max_questions: int = None,
    max_nonthinking_accuracy: Optional[float] = None,
):
    # Use defaults if not specified
    if max_questions is None:
        max_questions = DEFAULT_MAX_QUESTIONS
    if max_nonthinking_accuracy is None:
        max_nonthinking_accuracy = DEFAULT_MAX_NONTHINKING_ACCURACY

    # Get the parameter-specific output directory
    thinking_output_dir = get_parameter_specific_output_dir(
        max_questions, model_thinking, max_nonthinking_accuracy
    )

    print("Starting MMLU Thinking Mode Analysis")
    print(f"Configuration:")
    print(f"  Max questions: {max_questions}")
    print(f"  Max non-thinking accuracy: {max_nonthinking_accuracy}")
    print(f"  Output directory: {thinking_output_dir}")

    # Load data
    print("\n1. Loading data...")
    nonthinking_df = load_nonthinking_results(
        model_base, max_questions=max_questions
    )
    thinking_results = load_thinking_results(thinking_output_dir)

    if len(nonthinking_df) == 0:
        print("Error: No non-thinking results found")
        return

    # Combine results
    print("\n2. Combining results...")
    combined_df = combine_results(nonthinking_df, thinking_results)
    return combined_df


def main(
    model_base,
    model_thinking,
    max_questions: int = None,
    max_nonthinking_accuracy: Optional[float] = None,
):
    """Main analysis pipeline.

    Args:
        max_questions: Number of questions that were processed (default: DEFAULT_MAX_QUESTIONS)
        max_nonthinking_accuracy: Accuracy threshold that was used (default: DEFAULT_MAX_NONTHINKING_ACCURACY)
    """

    combined_df = get_combined_df(
        model_base, model_thinking, max_questions, max_nonthinking_accuracy
    )

    output_dir = "results/thinking_vs_nonthinking"
    model_str_base = model_base.split("/")[-1]
    model_str_thinking = model_thinking.split("/")[-1]
    output_dir = os.path.join(
        output_dir, "Base_" + model_str_base + "-Think_" + model_str_thinking
    )

    # Save combined results
    output_csv = os.path.join(output_dir, "combined_analysis.csv")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    print(f"\n3. Saving combined results to {output_csv}")
    combined_df.to_csv(output_csv, index=False)
    print(
        f"   Saved {len(combined_df)} rows with {len(combined_df.columns)} columns"
    )

    # Perform analysis
    print("\n4. Analyzing results...")
    analysis = analyze_results(combined_df)

    # Save analysis JSON
    analysis_json = os.path.join(output_dir, "analysis_report.json")
    # Convert numpy types to Python types for JSON serialization
    analysis_serializable = {}
    for key, value in analysis.items():
        if isinstance(value, (np.integer, np.floating)):
            analysis_serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            analysis_serializable[key] = value.tolist()
        elif isinstance(value, dict):
            analysis_serializable[key] = {
                k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                for k, v in value.items()
            }
        else:
            analysis_serializable[key] = value

    with open(analysis_json, "w") as f:
        json.dump(analysis_serializable, f, indent=2, default=str)
    print(f"   Analysis report saved to {analysis_json}")

    # Print report
    print_analysis_report(analysis)

    # Create visualizations
    print("\n5. Creating visualizations...")

    create_visualizations(combined_df, model_base, model_thinking, output_dir)

    print("\n✅ Analysis complete!")
    print(f"   Combined CSV: {output_csv}")
    print(f"   Analysis JSON: {analysis_json}")
    print(f"   Visualizations: {output_dir}/")

    return combined_df, analysis


if __name__ == "__main__":
    MODEL_BASE = "qwen3-30b-a3b"
    MODEL_THINKING = "qwen/qwen3-30b-a3b"

    df, analysis = main(
        MODEL_BASE,
        MODEL_THINKING,
        max_questions=16000,
        max_nonthinking_accuracy=None,
    )
