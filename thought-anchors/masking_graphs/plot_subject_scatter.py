from generate_graph_data import get_graph_fp
import pandas as pd
from constants import (
    SUBJECT2DOMAIN,
    CATEGORY_COLORS,
    STEM_LOGIC_SUBJECTS,
    LIFE_SCIENCES_SUBJECTS,
    MATH_PHYSICS_LOGIC_NAME,
    LIFE_SCIENCES_NAME,
    HUMANITIES_SOCIAL_NAME,
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


def darken_color(hex_color, factor=0.5):
    """
    Darken a hex color by blending it with black.
    factor=0.5 means halfway between original and black.
    """
    # Convert hex to RGB
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    # Blend with black (0, 0, 0)
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    # Convert back to hex
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_accuracy_kl_scatter_clean(
    df,
    metric="thinking_accuracy",
    kl_type="local",
    require_correct=-1,
    model_name="model",
):
    """
    Clean scatter plot without text labels, matching box plot styling.
    """
    # Calculate subject-level statistics for all metrics
    subject_stats = (
        df.groupby("subject")
        .agg(
            {
                "nonthinking_accuracy": "mean",
                "thinking_accuracy": "mean",
                "thinking_improvement": "mean",
                "local_kl_avg": "mean",
                "long_kl_avg": "mean",
            }
        )
        .reset_index()
    )

    subject_counts = df.groupby("subject").size().to_dict()
    subject_stats["count"] = subject_stats["subject"].map(subject_counts)

    # Match box plot figure size
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Use imported subject category constants
    stem_logic_subjects = STEM_LOGIC_SUBJECTS
    life_sciences_subjects = LIFE_SCIENCES_SUBJECTS

    # Define colors for categories
    category_colors = CATEGORY_COLORS

    # Assign colors based on subject category
    colors = []
    categories = []
    for subject in subject_stats["subject"]:
        if subject in stem_logic_subjects:
            colors.append(category_colors[MATH_PHYSICS_LOGIC_NAME])
            categories.append(MATH_PHYSICS_LOGIC_NAME)
        elif subject in life_sciences_subjects:
            colors.append(category_colors[LIFE_SCIENCES_NAME])
            categories.append(LIFE_SCIENCES_NAME)
        else:
            colors.append(category_colors[HUMANITIES_SOCIAL_NAME])
            categories.append(HUMANITIES_SOCIAL_NAME)

    subject_stats["category"] = categories

    kl_col = f"{kl_type}_kl_avg"

    # Create scatter plot by category (for legend)
    for category, color in category_colors.items():
        cat_data = subject_stats[subject_stats["category"] == category]
        if len(cat_data) > 0:
            ax.scatter(
                cat_data[metric],
                cat_data[kl_col],
                s=50,  # cat_data["count"] * 0.5 + 50,  # Size based on count
                c=color,
                label=f"{category}",
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

    # Remove top and right spines, make left and bottom black
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["bottom"].set_color("black")

    # NO TEXT LABELS - keeping plot clean

    # Calculate Spearman correlation and add trend line
    from scipy import stats

    # Calculate Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(
        subject_stats[metric], subject_stats[kl_col], nan_policy="omit"
    )

    # Drop NaN values for linear regression
    valid_mask = ~(subject_stats[metric].isna() | subject_stats[kl_col].isna())
    valid_x = subject_stats[metric][valid_mask]
    valid_y = subject_stats[kl_col][valid_mask]

    # Plot linear trend line
    if len(valid_x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_x, valid_y
        )
    else:
        slope, intercept, r_value, p_value, std_err = 0, 0, 0, 1, 0

    x_trend = np.linspace(
        subject_stats[metric].min(), subject_stats[metric].max(), 100
    )
    y_trend = slope * x_trend + intercept
    if spearman_p < 0.001:
        p_text = "p < .001"
    else:
        p_text = f"p = {spearman_p:.3f}"
    if r_value < 0:
        r_text = "r = -" + f"{r_value:.2f}"[2:]
    else:
        r_text = "r = " + f"{r_value:.2f}"[1:]
    ax.plot(
        x_trend,
        y_trend,
        "k--",
        alpha=0.5,
        linewidth=2,
        label=f"Trend (${r_text}, {p_text}$)",
    )

    metric2label = {
        "nonthinking_accuracy": "Non-thinking Accuracy",
        "thinking_accuracy": "Reasoning Accuracy",
    }

    metric_label = metric2label.get(metric, metric)

    kl_label = "Mean close-range effect" if kl_type == "local" else "Mean long-range effect"

    # Formatting to match box plot font sizes
    ax.set_xlabel(metric_label, fontsize=14, fontweight="medium", labelpad=10)
    ax.set_ylabel(kl_label, fontsize=14, fontweight="medium", labelpad=10)
    # ax.set_title(
    #     f"Subject Performance: {metric_label} vs {kl_label}",
    #     fontsize=14,
    #     fontweight="bold",
    # )

    # Match box plot tick settings
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12,
        direction="out",
        length=4,
        width=1,
    )

    # Format x-axis as percentages if it's an accuracy metric
    if "accuracy" in metric:
        import matplotlib.ticker as mticker

        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Make tick labels bold (after formatting)
    plt.draw()  # Force creation of tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("medium")

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Legend to match box plot style
    ax.legend(
        loc="best", fontsize=10, frameon=True, fancybox=False, shadow=False
    )

    # Set axis limits with padding
    x_padding = (
        subject_stats[metric].max() - subject_stats[metric].min()
    ) * 0.08
    y_padding = (
        subject_stats[kl_col].max() - subject_stats[kl_col].min()
    ) * 0.08
    ax.set_xlim(
        subject_stats[metric].min() - x_padding,
        subject_stats[metric].max() + x_padding,
    )
    ax.set_ylim(
        subject_stats[kl_col].min() - y_padding,
        subject_stats[kl_col].max() + y_padding,
    )

    # Save figure
    metric_short = (
        metric.replace("_", "")
        .replace("accuracy", "acc")
        .replace("improvement", "imp")
    )
    model_dir = f"plots"
    os.makedirs(model_dir, exist_ok=True)
    output_path = f"{model_dir}/{metric_short}_vs_{kl_type}_kl_clean_rc{require_correct}_{model_name}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # plt.show(block=True)
    plt.close()
    print(f"Saved clean scatter plot to {output_path}")


def avg_short_long_range(
    df, local_low=1, local_high=2, long_low=16, long_high=64
):
    local_k_cols = [f"total_kl_k{k}" for k in range(local_low, local_high + 1)]
    df["local_kl_avg"] = df[local_k_cols].mean(axis=1)

    long_k_cols = [f"total_kl_k{k}" for k in range(long_low, long_high + 1)]
    df["long_kl_avg"] = df[long_k_cols].mean(axis=1)
    return df


def load_graph_df(require_correct=True, model="qwen3-30b-a3b"):
    fp_out, model_name = get_graph_fp(model, require_correct)
    df = pd.read_csv(fp_out)
    df["domain"] = df["subject"].map(SUBJECT2DOMAIN)

    # df_kl, k_cols, k_values = get_df_kl(df)
    # print(df["total_kl_k1"])
    # quit()
    df = avg_short_long_range(df)
    df["thinking_accuracy"] = df["thinking_is_correct"].astype(
        float
    )  # Internal name stays the same
    df["nonthinking_accuracy"] = df["is_correct"].astype(float)
    return df


if __name__ == "__main__":
    REQUIRE_CORRECT = True
    model_name = "qwen3-30b-a3b"

    df = load_graph_df(require_correct=REQUIRE_CORRECT, model="qwen3-30b-a3b")
    # print(f'{df["subject"].nunique()=}')
    # print(f'{df["domain"].nunique()=}')
    # quit()

    # df_incorrect_unpaired = load_graph_df(
    #     require_correct=False, model="qwen3-30b-a3b"
    # )

    # hashes = set(df["hash_key"].unique())

    # df_incorrect = df_incorrect_unpaired[
    #     df_incorrect_unpaired["hash_key"].isin(hashes)
    # ]

    # df["local_kl_avg"] = (df["local_kl_avg"] + df_incorrect["local_kl_avg"]) / 2
    # df["long_kl_avg"] = (df["long_kl_avg"] + df_incorrect["long_kl_avg"]) / 2

    # Use the clean version without text labels
    # You can change metric to "nonthinking_accuracy" or "thinking_accuracy"
    # and kl_type to "local" or "long"
    for metric in ["thinking_accuracy"]:  # , "nonthinking_accuracy"
        for kl_type in [
            "local",
            "long",
        ]:  # Add "long" if you want long-range too
            plot_accuracy_kl_scatter_clean(
                df,
                require_correct=REQUIRE_CORRECT,
                metric=metric,
                kl_type=kl_type,
                model_name=model_name,
            )
    quit()
