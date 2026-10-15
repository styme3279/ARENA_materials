import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from plot_one_attn_matrix import plot_one_attn_mtx
from pytorch_models.model_config import model2layers_heads
from attention_analysis.receiver_head_funcs import (
    get_problem_vert_scores,
    get_top_k_receiver_heads,
)


def plot_dozen_layer_heads(
    target_layer=None,
    highlight_head=6,
    problem_ci=(1591, True),
    key_color="navy",
    top_k=None,
    model_name="qwen-14b",
    proximity_ignore=4,
    control_depth=False,
):
    assert not (target_layer is not None and top_k is not None)
    assert target_layer is None or top_k is None

    if top_k is not None:
        coords = get_top_k_receiver_heads(
            model_name=model_name,
            top_k=top_k,
            proximity_ignore=proximity_ignore,
            control_depth=control_depth,
        )
    else:
        num_heads = model2layers_heads(model_name)[1]
        coords = np.array([(target_layer, i) for i in range(num_heads)])

    target_layer_head_vert_scores = get_problem_vert_scores(
        coords,
        problem_ci,
        model_name=model_name,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
    )
    fig = plt.figure(figsize=(8, 3))

    if top_k is not None:
        plt.rcParams["font.size"] = 14
    else:
        plt.rcParams["font.size"] = 11
    if top_k is not None:
        plt.title(f"   Top {top_k} receiver heads")
    else:
        plt.title(f"Layer: {coords[0][0]}")

    for i, (layer, head) in enumerate(coords):
        vert_scores = target_layer_head_vert_scores[i, :]
        if top_k:
            if np.nanmax(vert_scores) > 0.01:
                print(f"Too high: {layer=}, {head=}, {np.nanmax(vert_scores)=}")
                continue
        if head == highlight_head and not top_k:
            plt.plot(
                vert_scores,
                label=f"Head: {head}",
                color=key_color,
                zorder=100,
                linewidth=1,
            )
        else:
            plt.plot(vert_scores, label=f"Head: {head}", linewidth=1)

    fp_out = f"plots/head_distributions/pn_{problem_ci[0]}-{problem_ci[1]}_head_{target_layer}-{highlight_head}_k{top_k}_{model_name}.png"
    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
    if top_k is not None:
        plt.ylim(0, 0.009)
        plt.yticks([0, 0.002, 0.004, 0.006, 0.008])
    elif target_layer == 36:
        plt.ylim(0, 0.007)
    else:
        plt.ylim(0, 0.007)
        # plt.ylim(0, 0.149)
    if top_k is not None:
        plt.ylabel("Receiver head score", fontsize=16, labelpad=11)
    else:
        plt.ylabel("Vertical attention score", fontsize=11, labelpad=7)
    plt.xlabel("Sentence position", fontsize=14 if top_k else 11, labelpad=7)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, -3))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().spines[["top", "right"]].set_visible(False)

    if top_k is not None:
        plt.xticks(np.arange(0, len(vert_scores), 10))
        plt.xlim(0, np.sum(~np.isnan(vert_scores)) - 1)
    else:
        plt.xticks(np.arange(0, len(vert_scores), 25))
        plt.xlim(0, len(vert_scores) - 1)
    plt.subplots_adjust(bottom=0.20 if top_k is not None else 0.15, top=0.8, left=0.15, right=0.95)

    plt.savefig(fp_out, dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot attention head distributions")
    parser.add_argument("--problem-num", type=int, default=1591, help="Problem number to analyze")
    parser.add_argument("--correct", action="store_true", default=True, help="Use correct solution")
    parser.add_argument("--layer", type=int, default=20, help="Target layer to plot")
    parser.add_argument("--highlight-head", type=int, default=0, help="Head to highlight in the plot")
    parser.add_argument("--top-k", type=int, default=None, help="Plot top K receiver heads instead of single layer")
    parser.add_argument("--model-name", type=str, default="qwen-15b", help="Model name")
    parser.add_argument("--proximity-ignore", type=int, default=4, help="Proximity ignore for vertical scores")
    parser.add_argument("--control-depth", action="store_true", help="Control for depth in vertical scores")
    parser.add_argument("--key-color", type=str, default="navy", help="Color for highlighted head")
    parser.add_argument("--output-dir", type=str, default="plots/head_distributions", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    parser.add_argument("--also-plot-matrix", action="store_true", help="Also plot attention matrix")
    parser.add_argument("--figsize", type=float, nargs=2, default=[8, 3], help="Figure size (width height)")
    
    args = parser.parse_args()
    
    # Set target_layer to None if using top_k
    target_layer = None if args.top_k else args.layer
    
    plot_dozen_layer_heads(
        problem_ci=(args.problem_num, args.correct),
        target_layer=target_layer,
        highlight_head=args.highlight_head,
        top_k=args.top_k,
        model_name=args.model_name,
        proximity_ignore=args.proximity_ignore,
        control_depth=args.control_depth,
        key_color=args.key_color,
    )
    
    if args.also_plot_matrix:
        plot_one_attn_mtx(
            problem_num=args.problem_num,
            is_correct=args.correct,
            layer=target_layer,
            head=args.highlight_head,
            top_k=args.top_k,
            model_name=args.model_name,
        )
