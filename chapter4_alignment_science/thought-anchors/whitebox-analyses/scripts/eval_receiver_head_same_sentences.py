import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.receiver_head_funcs import (
    get_3d_ar_kurtosis,
    get_all_problems_vert_scores,
    get_top_k_receiver_heads,
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
    parser.add_argument("--top-k", type=int, default=32, help="Top-k for the receiver heads")

    args = parser.parse_args()
    
    resp_layer_head_verts, _ = get_all_problems_vert_scores(
        model_name=args.model_name,
        proximity_ignore=args.proximity_ignore,
        control_depth=args.control_depth,
    )

    top_k_layer_head = get_top_k_receiver_heads(
        model_name=args.model_name,
        top_k=args.top_k,
        proximity_ignore=args.proximity_ignore,
        control_depth=args.control_depth,
    )

    M_corrs = []
    for i in range(len(resp_layer_head_verts)):
        layer_head_verts = resp_layer_head_verts[i]
        vert_scores = layer_head_verts[top_k_layer_head[:, 0], top_k_layer_head[:, 1]]
        vert_scores = vert_scores[:, :args.proximity_ignore] # drop nans at end
        resp_i_corrs = np.corrcoef(vert_scores)
        resp_i_corrs[np.diag_indices_from(resp_i_corrs)] = np.nan
        M_corr = np.nanmean(resp_i_corrs)
        if np.isnan(M_corr): # CoT is smaller than proximity ignore
            continue
        M_corrs.append(M_corr)

    GM_corrs = np.mean(M_corrs)
    print(f'Overall mean correlation: {GM_corrs:.3f}')
    
    
    num_layers = 48
    num_heads = 40
    all_heads = np.mgrid[0:num_layers, 0:num_heads].reshape(2, -1).T
    # Equivalent to: 
    # top_k_layer_head = []
    # for i in range(num_layers):
    #     for k in range(num_heads):
    #         top_k_layer_head.append([i, k])
    # all_heads = np.array(top_k_layer_head)

    M_corrs = []
    for i in range(len(resp_layer_head_verts)):
        layer_head_verts = resp_layer_head_verts[i]
        vert_scores = layer_head_verts[all_heads[:, 0], all_heads[:, 1]]
        vert_scores = vert_scores[:, :args.proximity_ignore] # drop nans at end
        resp_i_corrs = np.corrcoef(vert_scores)
        resp_i_corrs[np.diag_indices_from(resp_i_corrs)] = np.nan
        M_corr = np.nanmean(resp_i_corrs)
        if np.isnan(M_corr): # CoT is smaller than proximity ignore
            continue
        M_corrs.append(M_corr)


    GM_corrs = np.mean(M_corrs)
    print(f'All-head mean correlation: {GM_corrs:.3f}')
