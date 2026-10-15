import os
import sys
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.attn_supp_funcs import get_suppression_KL_matrix


def plot_supp_matrix_nice(
    sentence_sentence_scores0,
    model_name,
    problem_num,
    only_pre_convergence,
    quantiles=(0.05, 0.95),
):
    plot_dir = rf"plot_suppression/{model_name}/{only_pre_convergence}"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    trils = np.tril_indices_from(sentence_sentence_scores0, k=-11)
    sentence_scores = sentence_sentence_scores0[trils]

    vmin = np.nanquantile(sentence_scores, quantiles[0])
    vmax = np.nanquantile(sentence_scores, quantiles[1])

    plt.title(f"Reasoning model: {problem_num}")
    cmap = white_to_reds()
    plt.imshow(sentence_sentence_scores0, vmin=vmin, vmax=vmax, cmap=cmap)


def get_white_reds():
    colors = [(1, 1, 1), (1, 0, 0)]  # White to Red
    cmap_name = "white_to_red"
    white_red_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)
    return white_red_cmap


# Create a custom function to blend white with Blues
def white_to_reds(N=256):
    reds = plt.cm.Reds

    reds_colors = reds(np.linspace(0, 1, N))

    # Create an array that transitions from white to the Blues colormap
    white_color = np.array([1, 1, 1, 1])  # RGBA for white
    white_color = get_white_reds()(np.linspace(0, 1, N))

    # Create weights for blending (from 1 to 0 for white, from 0 to 1 for Blues)
    white_weights = np.linspace(1, 0, N)[:, np.newaxis]
    reds_weights = 1 - white_weights

    colors = white_weights * white_color + reds_weights * reds_colors

    return mcolors.LinearSegmentedColormap.from_list("WhiteToReds", colors)


def plot_single_problem(problem_num, only_pre_convergence="semi", take_log=True, figsize=(6, 6)):
    """
    Plot a single problem's suppression matrix.

    Args:
        problem_num: The problem number to plot
        only_pre_convergence: Whether to only include pre-convergence steps
        take_log: Whether to take log of values
        figsize: Figure size as (width, height) tuple
    """
    model_name = "qwen-14b"
    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95")
    layers_to_mask = {i: list(range(40)) for i in range(48)}

    plt.rcParams["font.size"] = 11
    plt.figure(figsize=figsize)

    # Note: layers_to_mask, quantize options, problem_dir, and only_pre_convergence
    # are no longer supported after refactoring
    sentence_sentence_scores = get_suppression_KL_matrix(
        problem_num=problem_num,
        p_nucleus=0.9999,
        model_name=model_name,
        is_correct=True,  # Assuming correct solutions
        only_first=None,
        take_log=take_log,
    )

    sentence_sentence_scores[np.triu_indices_from(sentence_sentence_scores, k=0)] = np.nan

    for i in range(sentence_sentence_scores.shape[0]):
        sentence_sentence_scores[i, :] -= np.nanmean(sentence_sentence_scores[i, :])
    sentence_sentence_scores[np.diag_indices_from(sentence_sentence_scores)] = np.nanquantile(
        sentence_sentence_scores, 0.99
    )

    plot_supp_matrix_nice(
        sentence_sentence_scores,
        model_name,
        problem_num,
        only_pre_convergence,
        quantiles=(0.5, 0.998),
    )

    plt.tick_params(axis="both", labelsize=11)
    xticks = np.arange(0, sentence_sentence_scores.shape[1], 10)
    plt.xticks(xticks)
    yticks = np.arange(0, sentence_sentence_scores.shape[0], 10)
    plt.yticks(yticks)
    plt.gca().spines[["top", "right"]].set_visible(False)

    pn = str(problem_num)
    is_correct = pn[-1] == "1"
    if is_correct:
        pn_title = f"Problem: {pn[:-2]} (correct)"
    else:
        pn_title = f"Problem: {pn[:-2]} (incorrect)"
    plt.title(pn_title)

    plt.xlabel("Sentence position", fontsize=14)
    plt.ylabel("Sentence position", fontsize=14)

    plt.tight_layout()
    plt.savefig(
        f"plots/suppression_single_problem_{problem_num}_{model_name}_{only_pre_convergence}.png",
        dpi=600,
        transparent=True,
    )
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot suppression matrix for a problem")
    parser.add_argument("--problem-num", type=int, default=468201, help="Problem number to analyze")
    parser.add_argument("--model-name", type=str, default="qwen-14b", help="Model name")
    parser.add_argument("--correct", action="store_true", default=True, help="Use correct solution")
    parser.add_argument("--p-nucleus", type=float, default=0.9999, help="Nucleus sampling parameter")
    parser.add_argument("--take-log", action="store_true", default=True, help="Take log of KL values")
    parser.add_argument("--only-first", type=int, default=None, help="Only process first N tokens of each sentence")
    parser.add_argument("--figsize", type=float, nargs=2, default=[6, 6], help="Figure size (width height)")
    parser.add_argument("--quantile-min", type=float, default=0.5, help="Min quantile for color scale")
    parser.add_argument("--quantile-max", type=float, default=0.998, help="Max quantile for color scale")
    parser.add_argument("--dpi", type=int, default=600, help="DPI for saved figure")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory")
    parser.add_argument("--font-size", type=int, default=11, help="Font size for plot")
    parser.add_argument("--tick-interval", type=int, default=10, help="Interval between ticks")
    parser.add_argument("--normalize-rows", action="store_true", default=True, help="Normalize rows by subtracting mean")
    parser.add_argument("--transparent", action="store_true", default=True, help="Save with transparent background")
    
    args = parser.parse_args()
    
    plt.rcParams["font.size"] = args.font_size
    plt.figure(figsize=tuple(args.figsize))
    
    sentence_sentence_scores = get_suppression_KL_matrix(
        problem_num=args.problem_num,
        p_nucleus=args.p_nucleus,
        model_name=args.model_name,
        is_correct=args.correct,
        only_first=args.only_first,
        take_log=args.take_log,
    )
    
    sentence_sentence_scores[np.triu_indices_from(sentence_sentence_scores, k=0)] = np.nan
    
    if args.normalize_rows:
        for i in range(sentence_sentence_scores.shape[0]):
            sentence_sentence_scores[i, :] -= np.nanmean(sentence_sentence_scores[i, :])
        sentence_sentence_scores[np.diag_indices_from(sentence_sentence_scores)] = np.nanquantile(
            sentence_sentence_scores, 0.99
        )
    
    plot_supp_matrix_nice(
        sentence_sentence_scores,
        args.model_name,
        args.problem_num,
        "normalized" if args.normalize_rows else "raw",
        quantiles=(args.quantile_min, args.quantile_max),
    )
    
    plt.tick_params(axis="both", labelsize=args.font_size)
    xticks = np.arange(0, sentence_sentence_scores.shape[1], args.tick_interval)
    plt.xticks(xticks)
    yticks = np.arange(0, sentence_sentence_scores.shape[0], args.tick_interval)
    plt.yticks(yticks)
    plt.gca().spines[["top", "right"]].set_visible(False)
    
    if args.correct:
        pn_title = f"Problem: {args.problem_num} (correct)"
    else:
        pn_title = f"Problem: {args.problem_num} (incorrect)"
    plt.title(pn_title)
    
    plt.xlabel("Sentence position", fontsize=14)
    plt.ylabel("Sentence position", fontsize=14)
    
    plt.tight_layout()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path / f"suppression_matrix_{args.problem_num}_{args.model_name}.png",
        dpi=args.dpi,
        transparent=args.transparent,
    )
    plt.show()
