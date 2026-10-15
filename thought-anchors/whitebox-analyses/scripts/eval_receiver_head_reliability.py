import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.receiver_head_funcs import (
    get_3d_ar_kurtosis,
    get_all_problems_vert_scores,
)


def get_kurt_matrix(
    model_name="qwen-14b",
    proximity_ignore=4,
    control_depth=False,
):
    resp_layer_head_verts, _ = get_all_problems_vert_scores(
        model_name=model_name,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
    )

    resp_layer_head_kurts = []

    for i in range(len(resp_layer_head_verts)):
        layer_head_verts = resp_layer_head_verts[i]
        layer_head_kurts = get_3d_ar_kurtosis(layer_head_verts)
        assert np.sum(np.isnan(layer_head_kurts[1:, :])) == 0  # Allow nan in layer 0
        resp_layer_head_kurts.append(layer_head_kurts)
    resp_layer_head_kurts = np.array(resp_layer_head_kurts)
    resp_layer_head_kurts[:, 0, :] = np.nan  # ignore layer 0 (no interesting attention)
    return resp_layer_head_kurts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate receiver head reliability using split-half correlation")
    parser.add_argument("--model-name", type=str, default="qwen-14b", help="Model name")
    parser.add_argument("--proximity-ignore", type=int, default=16, help="Proximity ignore for vertical scores")
    parser.add_argument("--control-depth", action="store_true", help="Control for depth in vertical scores")
    parser.add_argument("--figsize", type=float, nargs=2, default=[3.5, 3], help="Figure size (width height)")
    parser.add_argument("--font-size", type=int, default=11, help="Font size for plot")
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha for scatter points")
    parser.add_argument("--marker-size", type=int, default=20, help="Size of scatter points")
    parser.add_argument("--output-dir", type=str, default="plots/kurt_plots", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    parser.add_argument("--xlim", type=float, nargs=2, default=[-1, 42], help="X-axis limits")
    parser.add_argument("--ylim", type=float, nargs=2, default=[-1, 42], help="Y-axis limits")
    parser.add_argument("--tick-interval", type=int, default=10, help="Interval between ticks")
    
    args = parser.parse_args()
    
    kurts = get_kurt_matrix(args.model_name, args.proximity_ignore, args.control_depth)

    layers = np.arange(1, kurts.shape[1])
    heads = np.arange(kurts.shape[2])

    n_pn = kurts.shape[0]
    pn_cutoff = n_pn // 2

    kurts_first_half = kurts[::2, :, :]
    kurts_second_half = kurts[1::2, :, :]

    kurts_first_l = []
    kurts_second_l = []

    for layer in layers:
        for head in heads:
            kurt_first_half = kurts_first_half[:, layer, head]
            kurt_second_half = kurts_second_half[:, layer, head]

            kurts_first_l.append(np.mean(kurt_first_half))
            kurts_second_l.append(np.mean(kurt_second_half))

    kurts_first_l = np.array(kurts_first_l)
    kurts_second_l = np.array(kurts_second_l)

    r, p = stats.pearsonr(kurts_first_l, kurts_second_l)
    print(f"Reliability: r={r:.2f}, p={p:.4f}")

    plt.rcParams["font.size"] = args.font_size

    fig = plt.figure(figsize=tuple(args.figsize))

    plt.scatter(kurts_first_l, kurts_second_l, color="dodgerblue", alpha=args.alpha, s=args.marker_size)

    # Add line of best fit - using fixed endpoints
    z = np.polyfit(kurts_first_l, kurts_second_l, 1)
    slope, intercept = z

    # Create line using the min and max of x-axis for consistent endpoints
    x_min, x_max = plt.xlim()
    x_line = np.array([x_min, x_max])
    y_line = slope * x_line + intercept

    plt.plot(x_line, y_line, "k--", alpha=0.7, linewidth=1.5)

    plt.text(
        0.6,
        0.7,
        f"r = {r:.2f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    plt.axis("square")
    plt.xlim(args.xlim)
    plt.ylim(args.ylim)
    plt.xticks(np.arange(args.xlim[0], args.xlim[1] + 1, args.tick_interval))
    plt.yticks(np.arange(args.ylim[0], args.ylim[1] + 1, args.tick_interval))
    plt.gca().spines[["top", "right"]].set_visible(False)

    plt.xlabel("First half kurtosis", labelpad=7)
    plt.ylabel("Second half kurtosis", labelpad=7)
    plt.title("Split-half reliability assessment", fontsize=12, pad=10)
    
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fp_out = output_dir / f"reliability_{args.model_name}_pi{args.proximity_ignore}.png"
    plt.subplots_adjust(bottom=0.17, top=0.8, left=0.1, right=0.95)
    plt.savefig(fp_out, dpi=args.dpi)
    plt.show()
