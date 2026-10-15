import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_receiver_head_reliability import get_kurt_matrix
from pytorch_models.model_config import model2layers_heads


def plot_kurt_data(args):
    """Plot kurtosis statistics for attention heads.

    Args:
        args: Command line arguments with model configuration
    """
    kurts = get_kurt_matrix(
        model_name=args.model_name,
        proximity_ignore=args.proximity_ignore,
        control_depth=args.control_depth,
    )

    kurt = np.median(kurts, axis=0)

    layer_l = []
    kurt_l = []
    for layer in range(kurt.shape[0]):
        for j in range(kurt.shape[1]):
            if np.isnan(kurt[layer, j]):
                continue
            kurt_l.append(kurt[layer, j])
            layer_l.append(layer)
    layer_l = np.array(layer_l)
    kurt_l = np.array(kurt_l)

    plt.rcParams["font.size"] = args.scatter_font_size

    fig = plt.figure(figsize=tuple(args.scatter_figsize))

    plt.scatter(
        layer_l, kurt_l, color=args.color, alpha=args.alpha, s=args.scatter_size
    )

    n_layers, _ = model2layers_heads(args.model_name)
    plt.xlim(0, n_layers)
    plt.xlabel("Layer", labelpad=7)
    plt.ylabel("Kurtosis", labelpad=7)
    plt.title(
        "Kurtosis of each attention head's\nvertical score", fontsize=12, pad=0
    )

    plt.gca().spines[["top", "right"]].set_visible(False)

    pi_str = f"_pi{args.proximity_ignore}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fp_out = output_dir / f"kurt_layer_scatter_{args.model_name}{pi_str}.png"
    plt.subplots_adjust(bottom=0.17, top=0.8, left=0.2, right=0.95)
    plt.savefig(fp_out, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()

    fig = plt.figure(figsize=tuple(args.hist_figsize))
    flat = kurt.flatten()
    plt.rcParams["font.size"] = args.hist_font_size
    plt.hist(
        flat,
        bins=args.hist_bins,
        color=args.color,
        range=tuple(args.hist_range),
    )
    plt.xlim(-1, None)
    plt.title(
        "Histogram of attention head\nvertical score kurtoses",
        fontsize=12,
        pad=0,
    )
    plt.ylabel("Count", labelpad=7)
    plt.xlabel("Kurtosis", labelpad=7)
    plt.xticks(np.arange(0, 42, 10))
    plt.gca().spines[["top", "right"]].set_visible(False)

    fp_plot = output_dir / f"kurt_hist_{args.model_name}{pi_str}.png"
    plt.subplots_adjust(bottom=0.17, top=0.8, left=0.25, right=0.95)
    plt.savefig(fp_plot, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot kurtosis statistics for attention heads"
    )

    # Model configuration
    parser.add_argument(
        "--model-name", type=str, default="qwen-14b", help="Model name"
    )

    # parser.add_argument(
    #     "--model-name", type=str, default="llama-8b", help="Model name"
    # )
    parser.add_argument(
        "--proximity-ignore",
        type=int,
        default=16,
        help="Proximity ignore for vertical scores",
    )
    parser.add_argument(
        "--control-depth", action="store_true", help="Control for depth"
    )

    # Scatter plot settings
    parser.add_argument(
        "--scatter-figsize",
        type=float,
        nargs=2,
        default=[4.5, 3.5],
        help="Figure size for scatter plot (width height)",
    )
    parser.add_argument(
        "--scatter-font-size",
        type=int,
        default=11,
        help="Font size for scatter plot",
    )
    parser.add_argument(
        "--scatter-size", type=int, default=20, help="Size of scatter points"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.25, help="Alpha for scatter points"
    )
    parser.add_argument(
        "--color", type=str, default="dodgerblue", help="Color for plots"
    )

    # Histogram settings
    parser.add_argument(
        "--hist-figsize",
        type=float,
        nargs=2,
        default=[3, 3.5],
        help="Figure size for histogram (width height)",
    )
    parser.add_argument(
        "--hist-font-size", type=int, default=12, help="Font size for histogram"
    )
    parser.add_argument(
        "--hist-bins", type=int, default=80, help="Number of bins for histogram"
    )
    parser.add_argument(
        "--hist-range",
        type=float,
        nargs=2,
        default=[0, 40],
        help="Range for histogram",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/kurt_plots",
        help="Output directory",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for saved figures"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively",
        default=False,
    )

    args = parser.parse_args()

    plot_kurt_data(args)
