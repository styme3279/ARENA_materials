from generate_graph_data import get_graph_fp
import pandas as pd
from constants import (
    SUBJECT2DOMAIN,
    CATEGORY_COLORS,
    STEM_LOGIC_SUBJECTS,
    LIFE_SCIENCES_SUBJECTS,
)
from resample.plot_suppression import (
    get_df_kl,
    plot_k_profiles_by_domain,
    create_accuracy_vs_kl_scatter,
    create_k_correlation_timeseries,
)
import matplotlib.pyplot as plt
import numpy as np
import os
from plot_subject_scatter import load_graph_df
from constants import (
    MATH_PHYSICS_LOGIC_NAME,
    LIFE_SCIENCES_NAME,
    HUMANITIES_SOCIAL_NAME,
)


def plot_kl_boxplot_by_domain(df, model_name="model", require_correct=-1):
    """
    Create box plots showing local and long-range KL effects for three domains using seaborn.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Set style
    sns.set_style("whitegrid")

    # Assign domains
    stem_logic_subjects = STEM_LOGIC_SUBJECTS
    life_sciences_subjects = LIFE_SCIENCES_SUBJECTS

    domains = []
    for subject in df["subject"]:
        if subject in stem_logic_subjects:
            domains.append(MATH_PHYSICS_LOGIC_NAME)
        elif subject in life_sciences_subjects:
            domains.append(LIFE_SCIENCES_NAME)
        else:
            domains.append(HUMANITIES_SOCIAL_NAME)

    df["domain"] = domains

    # Create separate dataframes for local and long-range
    df_local = df[["domain", "local_kl_avg"]].copy()
    df_local["KL Type"] = "Local (k=1-5)"
    df_local["value"] = df_local["local_kl_avg"]

    df_long = df[["domain", "long_kl_avg"]].copy()
    df_long["KL Type"] = "Long-range (k=20-50)"
    df_long["value"] = df_long["long_kl_avg"]

    # Combine them
    df_plot = pd.concat(
        [
            df_local[["domain", "KL Type", "value"]],
            df_long[["domain", "KL Type", "value"]],
        ]
    )

    # Remove top 5 outliers from each KL type
    for kl_type in ["Local (k=1-5)", "Long-range (k=20-50)"]:
        mask = df_plot["KL Type"] == kl_type
        values = df_plot.loc[mask, "value"]
        threshold = values.nlargest(5).min()
        df_plot = df_plot[
            (df_plot["KL Type"] != kl_type) | (df_plot["value"] < threshold)
        ]

    # Debug: Check the data
    print("\nData summary (after removing top 5 outliers):")
    print(f"Total rows in df_plot: {len(df_plot)}")
    print(df_plot.groupby(["domain", "KL Type"]).size())
    print("\nValue ranges:")
    stats_df = df_plot.groupby(["domain", "KL Type"])["value"].agg(
        ["mean", "std", "min", "max"]
    )
    print(stats_df)

    # Check scale difference
    local_mean = df_plot[df_plot["KL Type"] == "Local (k=1-5)"]["value"].mean()
    long_mean = df_plot[df_plot["KL Type"] == "Long-range (k=20-50)"][
        "value"
    ].mean()
    scale_ratio = local_mean / long_mean if long_mean > 0 else float("inf")
    print(f"\nScale ratio (local/long-range): {scale_ratio:.1f}x")

    # Create figure with two subplots - THINNER figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))

    # Define colors (already uses constants but just redefinined - can use CATEGORY_COLORS directly)
    palette = CATEGORY_COLORS

    domain_order = [
        MATH_PHYSICS_LOGIC_NAME,
        LIFE_SCIENCES_NAME,
        HUMANITIES_SOCIAL_NAME,
    ]
    # Create labels with line breaks for compactness
    domain_labels = [
        MATH_PHYSICS_LOGIC_NAME.replace(", ", ",\n").replace(" & ", " &\n"),
        LIFE_SCIENCES_NAME.replace(" ", "\n"),
        HUMANITIES_SOCIAL_NAME.replace("/", "/\n"),
    ]

    # Left subplot: Local KL
    df_local_plot = df_plot[df_plot["KL Type"] == "Local (k=1-5)"]
    sns.boxplot(
        data=df_local_plot,
        x="domain",
        y="value",
        palette=palette,
        order=domain_order,
        ax=ax1,
        width=0.5,  # Make boxes thinner
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6,
        },
    )
    ax1.set_title(
        "Close-range ($k$ = 1 or 2)", fontsize=14
    )  # , fontweight="bold")
    ax1.set_xlabel("")  # Remove domain label
    ax1.set_ylabel(
        "Mean causal effect", fontsize=14, labelpad=10
    )  # , fontweight="medium")
    # Set ticks to project outward for both x and y axes
    ax1.tick_params(axis="x", labelsize=11, direction="out", length=4, width=1)
    ax1.tick_params(axis="y", labelsize=11, direction="out", length=4, width=1)
    ax1.set_xticklabels(domain_labels)  # Use two-line labels
    ax1.set_ylim(bottom=0)  # Start at 0

    # Remove top and right spines, make left and bottom black
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.5)
    ax1.spines["left"].set_color("black")
    ax1.spines["bottom"].set_linewidth(1.5)
    ax1.spines["bottom"].set_color("black")

    # Add sample sizes
    for i, domain in enumerate(domain_order):
        n = len(df[df["domain"] == domain])
        y_max = ax1.get_ylim()[1]
        # ax1.text(i, y_max * 0.95, f"n={n}", ha="center", fontsize=10, fontweight="medium")

    # Right subplot: Long-range KL
    df_long_plot = df_plot[df_plot["KL Type"] == "Long-range (k=20-50)"]
    sns.boxplot(
        data=df_long_plot,
        x="domain",
        y="value",
        palette=palette,
        order=domain_order,
        ax=ax2,
        width=0.5,  # Make boxes thinner
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6,
        },
    )
    ax2.set_title(
        "Long-range ($k$ = 16-64)", fontsize=14
    )  # , fontweight="bold")
    ax2.set_xlabel("")  # Remove domain label
    ax2.set_ylabel("")  # Remove duplicate y-label
    # Set ticks to project outward for both x and y axes
    ax2.tick_params(axis="x", labelsize=11, direction="out", length=4, width=1)
    ax2.tick_params(axis="y", labelsize=11, direction="out", length=4, width=1)
    ax2.set_xticklabels(domain_labels)  # Use two-line labels
    ax2.set_ylim(bottom=0)  # Start at 0

    # Remove top and right spines, make left and bottom black
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(1.5)
    ax2.spines["left"].set_color("black")
    ax2.spines["bottom"].set_linewidth(1.5)
    ax2.spines["bottom"].set_color("black")

    plt.tight_layout()

    # Save figure
    os.makedirs("plots", exist_ok=True)
    output_path = f"plots/kl_boxplot_by_domain_rc{require_correct}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show(block=True)
    print(f"Saved KL boxplot to {output_path}")


if __name__ == "__main__":
    REQUIRE_CORRECT = True
    MODEL_NAME = "qwen3-30b-a3b"

    df = load_graph_df(require_correct=REQUIRE_CORRECT, model="qwen3-30b-a3b")
    plot_kl_boxplot_by_domain(
        df, model_name=MODEL_NAME, require_correct=REQUIRE_CORRECT
    )
