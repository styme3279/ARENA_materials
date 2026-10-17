"""
Plot boxplots comparing receiver head scores across different taxonomic categories.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from pathlib import Path
import scipy.stats as stats
from typing import Optional, List
import argparse


def get_type2color():
    """Get mapping from taxonomic type to color."""
    type2color = {
        "plan_generation": "tab:red",
        "active_computation": "tab:green",
        "fact_retrieval": "tab:orange",
        "uncertainty_management": "tab:purple",
        "result_consolidation": "tab:blue",
        "self_checking": "tab:brown",
        "problem_setup": "tab:gray",
        "final_answer_emission": "tab:pink",
    }
    return type2color


def get_type2label():
    """Get mapping from taxonomic type to display label."""
    type2label = {
        "plan_generation": "Plan\nGeneration",
        "active_computation": "Active\nComputation",
        "fact_retrieval": "Fact\nRetrieval",
        "uncertainty_management": "Uncertainty\nMgmt.",
        "result_consolidation": "Result\nConsolidation",
        "self_checking": "Self\nChecking",
        "problem_setup": "Problem\nSetup",
        "final_answer_emission": "Final\nAnswer",
    }
    return type2label


def load_receiver_csv(
    csv_path: str, min_sentences_per_problem: int = 10
) -> pd.DataFrame:
    """Load and preprocess receiver head scores CSV."""
    df = pd.read_csv(csv_path)

    # Parse taxonomic labels (they're stored as pipe-separated strings)
    df["tags"] = df["taxonomic_labels"].apply(
        lambda x: x.split("|") if x != "none" else []
    )

    # Explode the dataframe so each row has a single tag
    df_expanded = []
    for _, row in df.iterrows():
        tags = row["tags"]
        if len(tags) == 0:
            tags = ["none"]
        for tag in tags:
            new_row = row.copy()
            new_row["tag"] = tag
            df_expanded.append(new_row)

    df = pd.DataFrame(df_expanded)

    # Filter out problems with too few sentences
    sentences_per_problem = df.groupby(["problem_number", "is_correct"]).size()
    valid_problems = sentences_per_problem[
        sentences_per_problem >= min_sentences_per_problem
    ].index
    df["problem_id"] = df.apply(
        lambda x: (x["problem_number"], x["is_correct"]), axis=1
    )
    df = df[df["problem_id"].isin(valid_problems)]

    return df


def plot_receiver_taxonomy(
    model_name: str = "qwen-15b",
    top_k: int = 16,
    csv_dir: str = "csvs",
    plot_type: str = "box",
    min_cnt: int = 10,
    random_ef: bool = True,
    output_dir: str = "plots",
    tags_to_plot: Optional[List[str]] = None,
    pre_convergence_only: bool = False,
    proximity_ignore: int = 4,
):
    """
    Plot receiver head scores by taxonomic category.

    Args:
        model_name: Model name used in CSV filename
        top_k: k value used in receiver head calculation
        csv_dir: Directory containing CSV files
        plot_type: Type of plot ("box", "violin", or "bar")
        min_cnt: Minimum count for inclusion
        random_ef: Whether to use random effects (average per problem)
        output_dir: Directory to save plots
        tags_to_plot: Specific tags to plot (if None, use default set)
        pre_convergence_only: If True, only include sentences before convergence
    """

    # Load the CSV file
    csv_path = (
        Path(csv_dir) / f"receiver_head_scores_all_{model_name}_k{top_k}_pi{proximity_ignore}.csv"
    )
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    df = load_receiver_csv(str(csv_path))

    # Filter for pre-convergence only if requested
    if pre_convergence_only:
        if "pre_convergence" in df.columns:
            df = df[df["pre_convergence"] == True]
            print(f"Filtered to {len(df)} pre-convergence sentences")
        else:
            print(
                "Warning: 'pre_convergence' column not found in CSV. Using all data."
            )

    # Define tags to plot
    if tags_to_plot is None:
        tags_to_plot = [
            "plan_generation",
            "fact_retrieval",
            "active_computation",
            "uncertainty_management",
            "result_consolidation",
        ]

    # Filter to only include specified tags
    df = df[df["tag"].isin(tags_to_plot)]

    # Get color and label mappings
    type2label = get_type2label()
    type2color = get_type2color()

    # Set up the plot
    plt.rcParams["font.size"] = 11
    plt.figure(figsize=(7, 2.5))

    if plot_type == "bar":
        # Bar plot with error bars
        Ms = []
        SE_s = []

        for tag in tags_to_plot:
            df_tag = df[df["tag"] == tag]

            if random_ef:
                # Average scores per problem
                vals = []
                for (pn, is_correct), df_pn in df_tag.groupby(
                    ["problem_number", "is_correct"]
                ):
                    if len(df_pn) < min_cnt:
                        continue
                    pn_mean = df_pn["receiver_head_score"].mean()
                    vals.append(pn_mean)
            else:
                vals = df_tag["receiver_head_score"].tolist()

            if len(vals) == 0:
                Ms.append(0)
                SE_s.append(0)
                continue

            M = np.nanmean(vals)
            SE = np.nanstd(vals) / np.sqrt(len(vals))
            t = M / SE if SE > 0 else 0
            N = len(vals)

            print(f"{tag}: M={M:.6f} (SE={SE:.6f}) t={t:.3f} N={N}")
            Ms.append(M)
            SE_s.append(SE)

        colors = [type2color.get(t, "gray") for t in tags_to_plot]
        labels = [type2label.get(t, t) for t in tags_to_plot]

        plt.bar(
            labels,
            Ms,
            yerr=np.array(SE_s) * 1.96,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            capsize=5,
            alpha=0.8,
        )

        # Color the x-tick labels
        for i, color in enumerate(colors):
            plt.gca().get_xticklabels()[i].set_color(color)

    else:  # box or violin plot
        # Prepare data for seaborn
        plot_data = []

        # Filter by minimum count per problem-tag pair
        if min_cnt > 0:
            cnt_pn_tag = (
                df.groupby(["problem_number", "is_correct", "tag"])
                .size()
                .reset_index(name="cnt")
            )
            df_bad = cnt_pn_tag[cnt_pn_tag["cnt"] <= min_cnt]
            bad_pairs = set(
                zip(
                    df_bad["problem_number"],
                    df_bad["is_correct"],
                    df_bad["tag"],
                )
            )
            df = df[
                ~df.apply(
                    lambda row: (
                        row["problem_number"],
                        row["is_correct"],
                        row["tag"],
                    )
                    in bad_pairs,
                    axis=1,
                )
            ]

        for tag in tags_to_plot:
            df_tag = df[df["tag"] == tag]

            if random_ef:
                # Average scores per problem
                for (pn, is_correct), df_pn in df_tag.groupby(
                    ["problem_number", "is_correct"]
                ):
                    if len(df_pn) < min_cnt:
                        continue
                    pn_mean = df_pn["receiver_head_score"].mean()
                    plot_data.append(
                        {
                            "tag": tag,
                            "value": pn_mean,
                            "problem_id": f"{pn}_{is_correct}",
                        }
                    )
            else:
                for _, row in df_tag.iterrows():
                    plot_data.append(
                        {
                            "tag": tag,
                            "value": row["receiver_head_score"],
                            "problem_id": f"{row['problem_number']}_{row['is_correct']}",
                        }
                    )

        if len(plot_data) == 0:
            print("No data to plot after filtering")
            return

        plot_df = pd.DataFrame(plot_data)

        # Statistical comparisons
        print("\nPairwise comparisons:")
        for i, tag0 in enumerate(tags_to_plot):
            for tag1 in tags_to_plot[i + 1 :]:
                df_tag0 = plot_df[plot_df["tag"] == tag0]
                df_tag1 = plot_df[plot_df["tag"] == tag1]

                # Find common problems for paired test
                common_problems = set(df_tag0["problem_id"]) & set(
                    df_tag1["problem_id"]
                )
                df_tag0_common = df_tag0[
                    df_tag0["problem_id"].isin(common_problems)
                ]
                df_tag1_common = df_tag1[
                    df_tag1["problem_id"].isin(common_problems)
                ]

                if len(df_tag0_common) > 1:
                    # Sort by problem_id to ensure matching
                    df_tag0_common = df_tag0_common.sort_values("problem_id")
                    df_tag1_common = df_tag1_common.sort_values("problem_id")

                    t, p = stats.ttest_rel(
                        df_tag0_common["value"].values,
                        df_tag1_common["value"].values,
                        nan_policy="omit",
                    )
                    print(
                        f"  {tag0} vs {tag1}: t={t:.2f}, p={p:.3f}, N={len(common_problems)}"
                    )

        # Add labels for plotting
        plot_df["label"] = plot_df["tag"].map(type2label)
        labels = [type2label.get(t, t) for t in tags_to_plot]
        colors = [type2color.get(t, "gray") for t in tags_to_plot]

        if plot_type == "box":
            ax = sns.boxplot(
                data=plot_df,
                x="label",
                y="value",
                order=labels,
                palette={
                    type2label.get(tag, tag): type2color.get(tag, "gray")
                    for tag in tags_to_plot
                },
                width=0.6,
            )
        else:  # violin
            ax = sns.violinplot(
                data=plot_df,
                x="label",
                y="value",
                order=labels,
                palette={
                    type2label.get(tag, tag): type2color.get(tag, "gray")
                    for tag in tags_to_plot
                },
            )

        # Color x-tick labels
        for i, color in enumerate(colors):
            if i < len(plt.gca().get_xticklabels()):
                plt.gca().get_xticklabels()[i].set_color(color)

    # Format plot
    plt.xlabel("")
    plt.ylabel("Mean receiver-head score")
    plt.xticks(fontsize=10)

    # Format y-axis with scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, -3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Remove top and right spines
    plt.gca().spines[["top", "right"]].set_visible(False)

    # Add title
    model_display = "Qwen-14B" if "qwen" in model_name.lower() else model_name
    plt.title(
        f"Receiver-head scores by sentence category",# ({model_display}, k={top_k})",
        fontsize=11,
    )
    plt.ylim(0, 0.00042)

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    suffix = "_pre_conv" if pre_convergence_only else ""
    fp_out = (
        output_path / f"receiver_taxonomy_{model_name}_k{top_k}{suffix}_pi{proximity_ignore}.png"
    )

    plt.subplots_adjust(bottom=0.2, top=0.85, left=0.12, right=0.95)
    plt.savefig(fp_out, dpi=300)
    print(f"\nPlot saved to {fp_out}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot receiver head scores by taxonomy"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen-14b",
        help="Model name (must match CSV filename)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Top-k value used in receiver head calculation",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="csvs",
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="box",
        choices=["box", "violin", "bar"],
        help="Type of plot to create",
    )
    parser.add_argument(
        "--min-cnt", type=int, default=2, help="Minimum count for inclusion"
    )
    parser.add_argument(
        "--random-ef",
        action="store_true",
        default=True,
        help="Use random effects (average per problem)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--pre-convergence-only",
        action="store_true",
        default=True,
        help="Only include sentences before model convergence",
    )
    parser.add_argument(
        "--proximity-ignore",
        type=int,
        default=16,
        help="Proximity ignore for vertical scores",
    )

    args = parser.parse_args()

    plot_receiver_taxonomy(
        model_name=args.model_name,
        top_k=args.top_k,
        csv_dir=args.csv_dir,
        plot_type=args.plot_type,
        min_cnt=args.min_cnt,
        random_ef=args.random_ef,
        output_dir=args.output_dir,
        pre_convergence_only=args.pre_convergence_only,
        proximity_ignore=args.proximity_ignore
    )
