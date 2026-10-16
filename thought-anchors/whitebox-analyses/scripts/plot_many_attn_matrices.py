import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.attn_funcs import get_avg_attention_matrix
from attention_analysis.receiver_head_funcs import (
    get_problem_text_sentences,
    get_top_k_receiver_heads,
    get_model_rollouts_root,
)
from pytorch_models.model_config import model2layers_heads


def white_to_blues(N=256):
    """Create a custom colormap that transitions from white to blues."""
    blues = plt.cm.Blues

    # Get Blues colormap array
    blues_colors = blues(np.linspace(0, 1, N))

    # Create white to blue transition
    colors = [(1, 1, 1), (0, 0, 1)]  # White to Blue
    white_blue_cmap = mcolors.LinearSegmentedColormap.from_list("white_to_blue", colors)
    white_color = white_blue_cmap(np.linspace(0, 1, N))

    # Create weights for blending
    white_weights = np.linspace(1, 0, N)[:, np.newaxis]
    blues_weights = 1 - white_weights

    # Blend the colors
    blended_colors = white_weights * white_color + blues_weights * blues_colors

    # Create a new colormap
    return mcolors.LinearSegmentedColormap.from_list("WhiteToBlues", blended_colors)


def get_problem_nums(model_name="qwen-14b", num_problems=20):
    """Get list of problem numbers from the rollouts directory."""
    dir_root = get_model_rollouts_root(model_name)

    # Get problem numbers from correct solutions directory
    correct_dir = os.path.join(dir_root, "correct_base_solution")
    if not os.path.exists(correct_dir):
        raise ValueError(f"Directory not found: {correct_dir}")

    problem_dirs = os.listdir(correct_dir)
    problem_nums = []

    for problem_dir in problem_dirs[:num_problems]:
        if problem_dir.startswith("problem_"):
            # Extract the problem number
            problem_num = int(problem_dir.replace("problem_", ""))
            # Add suffix for correct (1) or incorrect (0)
            problem_nums.append(f"{problem_num}1")  # Using correct solutions

    return problem_nums[:num_problems]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot grid of attention matrices for multiple problems")
    parser.add_argument("--model-name", type=str, default="qwen-15b", help="Model name")
    parser.add_argument("--num-problems", type=int, default=20, help="Number of problems to plot")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top receiver heads to identify")
    parser.add_argument("--proximity-ignore", type=int, default=4, help="Proximity ignore for receiver heads")
    parser.add_argument("--control-depth", action="store_true", help="Control for depth")
    parser.add_argument("--n-rows", type=int, default=5, help="Number of rows in grid")
    parser.add_argument("--n-cols", type=int, default=4, help="Number of columns in grid")
    parser.add_argument("--figsize", type=float, nargs=2, default=[7, 8.5], help="Figure size (width height)")
    parser.add_argument("--output-dir", type=str, default="plots/attn_matrices", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    parser.add_argument("--use-first-head", action="store_true", help="Use first receiver head from top-k")
    parser.add_argument("--layer", type=int, default=36, help="Default layer if no receiver heads found")
    parser.add_argument("--head", type=int, default=6, help="Default head if no receiver heads found")
    parser.add_argument("--quantile-min", type=float, default=0.05, help="Min quantile for color scale")
    parser.add_argument("--quantile-max", type=float, default=0.95, help="Max quantile for color scale")
    
    args = parser.parse_args()
    
    # Get problem numbers
    problem_nums = get_problem_nums(args.model_name, num_problems=args.num_problems)

    # Get top receiver heads
    coords = get_top_k_receiver_heads(
        model_name=args.model_name,
        top_k=args.top_k,
        proximity_ignore=args.proximity_ignore,
        control_depth=args.control_depth,
    )

    n_row = args.n_rows
    n_col = args.n_cols

    # Use the first receiver head from the top k
    if len(coords) > 0:
        target_layer, target_head = coords[0]
    else:
        target_layer, target_head = args.layer, args.head  # Default fallback

    # Create grid of targets
    idxs = np.arange(min(20, len(problem_nums)))
    idxs = np.reshape(idxs, (n_row, n_col))
    targets = [[None] * n_col for _ in range(n_row)]

    for i in range(n_row):
        for j in range(n_col):
            if idxs[i, j] < len(problem_nums):
                targets[i][j] = (target_layer, target_head, problem_nums[idxs[i, j]])

    head_all_same = True
    pn_all_same = False

    fig, axs = plt.subplots(len(targets), len(targets[0]), figsize=tuple(args.figsize))

    for row_idx, row in enumerate(targets):
        for col_idx, target in enumerate(row):
            if target is None:
                axs[row_idx, col_idx].axis("off")
                continue

            print(f"Processing: {row_idx}/{col_idx}")
            layer, head, pn = target

            # Extract problem number and correctness
            pn_str = str(pn)
            is_correct = pn_str[-1] == "1"
            problem_num = int(pn_str[:-1])

            # Get text and sentences
            text, sentences = get_problem_text_sentences(problem_num, is_correct, model_name=args.model_name)

            # Get averaged attention matrix
            avg_matrix = get_avg_attention_matrix(
                text,
                args.model_name,
                layer,
                head,
                sentences=sentences,
            )

            print(f"{avg_matrix.shape=}")

            # Remove first and last rows/columns (prompt and output)
            avg_matrix = avg_matrix[1:-1, 1:-1]

            # Get lower triangle for computing quantiles
            avg_matrix_tril = np.tril(avg_matrix, k=-4)
            vmin = np.nanquantile(avg_matrix_tril, args.quantile_min)
            vmax = np.nanquantile(avg_matrix_tril, args.quantile_max)

            # Create custom colormap
            white_blue_cmap = white_to_blues()

            # Plot the matrix
            axs[row_idx, col_idx].imshow(avg_matrix, vmin=0, vmax=vmax, cmap=white_blue_cmap)

            # Set title
            if pn_all_same:
                title = f"Layer: {layer}, Head: {head}"
            else:
                if is_correct:
                    pn_title = f"#{problem_num} (correct)"
                else:
                    pn_title = f"#{problem_num} (incorrect)"
                if not head_all_same:
                    title = f"{pn_title}\nLayer: {layer}, Head: {head}"
                else:
                    title = pn_title
            axs[row_idx, col_idx].set_title(title, fontsize=11 if head_all_same else 10)

            axs[row_idx, col_idx].tick_params(axis="both", labelsize=10)

            if avg_matrix.shape[0] > 124:
                xticks = np.arange(0, avg_matrix.shape[1], 50)
                yticks = np.arange(0, avg_matrix.shape[0], 50)
            elif avg_matrix.shape[0] > 30:
                xticks = np.arange(0, avg_matrix.shape[1], 25)
                yticks = np.arange(0, avg_matrix.shape[0], 25)
            else:
                xticks = np.arange(0, avg_matrix.shape[1], 10)
                yticks = np.arange(0, avg_matrix.shape[0], 10)

            # Format tick labels to have consistent width
            axs[row_idx, col_idx].set_xticks(xticks)
            axs[row_idx, col_idx].set_yticks(yticks)
            # axs[row_idx, col_idx].set_xticklabels([f"{x:>3}" for x in xticks])
            # axs[row_idx, col_idx].set_yticklabels([f"{y:>3}" for y in yticks])
            axs[row_idx, col_idx].set_xlim(0, avg_matrix.shape[1] - 1)
            axs[row_idx, col_idx].set_ylim(avg_matrix.shape[0] - 1, 0)

            # Remove top and right spines
            axs[row_idx, col_idx].spines["top"].set_visible(False)
            axs[row_idx, col_idx].spines["right"].set_visible(False)
            if row_idx == len(targets) - 1:
                axs[row_idx, col_idx].set_xlabel("Sentence position", fontsize=10)
            if col_idx == 0:
                axs[row_idx, col_idx].set_ylabel("Sentence position", fontsize=10, labelpad=5)

    if head_all_same:
        plt.suptitle(f"All responses for layer: {target_layer}, head: {target_head}", fontsize=13)
    elif pn_all_same:
        plt.suptitle(f"Attention weights for problem examples", fontsize=13)

    plt.tight_layout()
    fig.align_labels()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if head_all_same:
        plt.savefig(
            output_dir / f"receiver_head_{target_layer}_{target_head}_pn_examples.png", dpi=args.dpi
        )
    elif pn_all_same:
        plt.savefig(output_dir / f"receiver_pn_examples.png", dpi=args.dpi)
    plt.close()
