import os
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.attn_supp_funcs import get_suppression_KL_matrix
from attention_analysis.receiver_head_funcs import (
    get_problem_text_sentences,
    get_model_rollouts_root,
)


def plot_sentence_suppression_line(
    sentence_sentence_scores,
    sentence_num,
    problem_num,
    is_correct,
    output_path=None,
    show_plot=False,
):
    """
    Plot the suppression effect of a specific sentence on all other sentences.

    Args:
        sentence_sentence_scores: NxN matrix of KL divergences
        sentence_num: Which sentence was suppressed (column index)
        problem_num: Problem number for labeling
        is_correct: Whether this is a correct solution
        model_name: Model name for labeling
        output_path: Where to save the plot (None to not save)
        show_plot: Whether to display the plot

    Returns:
        None
    """
    # Extract the column corresponding to the suppressed sentence
    sentence_KL_logs = sentence_sentence_scores[:, sentence_num]

    # Calculate y-axis limits with padding
    valid_values = sentence_KL_logs[~np.isnan(sentence_KL_logs)]
    if len(valid_values) == 0:
        print(f"Warning: No valid KL values for sentence {sentence_num}")
        return

    kl_log_low = np.nanmin(sentence_KL_logs)
    kl_log_high = np.nanmax(sentence_KL_logs)

    # Add some padding to y-axis
    padding = (kl_log_high - kl_log_low) * 0.02
    if padding < 0.1:
        padding = 0.1
    y_min = kl_log_low - padding
    y_max = kl_log_high + padding

    # Create figure
    plt.rcParams["font.size"] = 11
    plt.figure(figsize=(6, 4))

    # Plot vertical line for suppressed sentence
    plt.vlines(
        sentence_num,
        y_min,
        y_max,
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"Suppressed sentence {sentence_num}",
    )

    # Plot the KL divergence values
    sentence_idxs = np.arange(len(sentence_KL_logs))
    plt.plot(
        sentence_idxs,
        sentence_KL_logs,
        marker=".",
        markersize=2,
        color="firebrick",
        linewidth=1,
    )

    # Highlight important points above threshold
    threshold = -6
    for i in range(len(sentence_idxs)):
        x_coord = sentence_idxs[i]
        y_coord = sentence_KL_logs[i]

        # Only highlight points above threshold and not NaN
        if not np.isnan(y_coord) and y_coord > threshold:
            # Add larger marker for highlighted points
            plt.plot(
                x_coord,
                y_coord,
                marker="o",
                markersize=6,
                color="maroon",
                linestyle="",
            )

            # Add text label (skip the suppressed sentence itself)
            if i != sentence_num:
                plt.text(
                    x_coord + 0.8,
                    y_coord + 0.2,
                    f"{x_coord}",
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    color="k",
                )

    # Set axis limits and labels
    plt.ylim(y_min, y_max * 1.06)
    plt.xlim(0, len(sentence_KL_logs) - 1)
    plt.ylabel("Mean Log(KL Divergence + 1e-9)", labelpad=7)
    plt.xlabel("Sentence Number", labelpad=7)

    # Create title
    correct_str = "correct" if is_correct else "incorrect"
    plt.title(f"Suppression effect: Problem {problem_num} ({correct_str})")

    # Style adjustments
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.95)

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Saved plot to {output_path}")

    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_sentence_importance(
    sentence_sentence_scores,
    threshold=-5.0,
    top_k=5,
):
    """
    Analyze which sentences are most important (highest KL when suppressed).

    Args:
        sentence_sentence_scores: NxN matrix of KL divergences
        threshold: Minimum KL value to consider "important"
        top_k: Number of top sentences to return

    Returns:
        Dictionary with analysis results
    """
    n_sentences = sentence_sentence_scores.shape[0]

    # Calculate statistics for each suppressed sentence
    sentence_stats = []
    for sent_idx in range(n_sentences):
        column = sentence_sentence_scores[:, sent_idx]
        valid_values = column[~np.isnan(column)]

        if len(valid_values) > 0:
            stats = {
                "sentence_idx": sent_idx,
                "mean_kl": np.mean(valid_values),
                "max_kl": np.max(valid_values),
                "std_kl": np.std(valid_values),
                "num_affected": np.sum(valid_values > threshold),
                "total_impact": np.sum(valid_values[valid_values > threshold]),
            }
            sentence_stats.append(stats)

    # Sort by mean KL impact
    sentence_stats.sort(key=lambda x: x["mean_kl"], reverse=True)

    return {
        "top_sentences": sentence_stats[:top_k],
        "all_stats": sentence_stats,
        "most_impactful_idx": sentence_stats[0]["sentence_idx"] if sentence_stats else None,
    }


def process_problem_suppression(
    problem_num,
    is_correct=True,
    model_name="qwen-14b",
    plot_top_k=3,
    output_dir="plots/suppression_analysis",
):
    """
    Process a problem and generate suppression plots for the most important sentences.

    Args:
        problem_num: Problem number to analyze
        is_correct: Whether to use correct or incorrect solution
        model_name: Model to use
        plot_top_k: Number of top important sentences to plot
        output_dir: Directory for saving plots

    Returns:
        Analysis results and KL matrix
    """
    print(f"\nProcessing problem {problem_num} ({'correct' if is_correct else 'incorrect'})...")

    sentence_sentence_scores = get_suppression_KL_matrix(
        problem_num=problem_num,
        p_nucleus=0.9999,
        model_name=model_name,
        is_correct=is_correct,
        take_log=True,
    )

    if sentence_sentence_scores is None:
        print(f"Failed to compute KL matrix for problem {problem_num}")
        return None, None

    print(f"KL matrix shape: {sentence_sentence_scores.shape}")

    # Analyze sentence importance
    analysis = analyze_sentence_importance(sentence_sentence_scores)

    print(f"\nTop {plot_top_k} most impactful sentences:")
    for i, stats in enumerate(analysis["top_sentences"][:plot_top_k]):
        print(
            f"  {i+1}. Sentence {stats['sentence_idx']}: "
            f"mean KL = {stats['mean_kl']:.3f}, "
            f"affects {stats['num_affected']} sentences"
        )

        # Plot this sentence's suppression effect
        output_path = (
            Path(output_dir)
            / model_name
            / f"problem_{problem_num}_{is_correct}"
            / f"suppress_sent_{stats['sentence_idx']}.png"
        )
        plot_sentence_suppression_line(
            sentence_sentence_scores,
            stats["sentence_idx"],
            problem_num,
            is_correct,
            model_name,
            output_path=output_path,
            show_plot=False,
        )

    return analysis, sentence_sentence_scores


if __name__ == "__main__":
    # Configuration
    model_name = "qwen-15b"
    problem_num = 4682  # Example problem
    is_correct = True
    plot_top_k = 3  # Plot top 3 most important sentences

    # Process single problem
    analysis, kl_matrix = process_problem_suppression(
        problem_num=problem_num,
        is_correct=is_correct,
        model_name=model_name,
        plot_top_k=plot_top_k,
        output_dir="plots/suppression_analysis",
    )

    if analysis:
        # You can also manually plot a specific sentence
        sentence_num = 32  # Example: plot suppression of sentence 32
        if kl_matrix is not None and sentence_num < kl_matrix.shape[0]:
            output_path = f"plots/suppress_sentence_{problem_num}_{sentence_num}.png"
            plot_sentence_suppression_line(
                kl_matrix,
                sentence_num,
                problem_num,
                is_correct,
                model_name,
                output_path=output_path,
                show_plot=True,  # Show this plot
            )
