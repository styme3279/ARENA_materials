"""Modified correlation timeseries plot with matching styling and within-subject option."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from constants import (
    STEM_LOGIC_SUBJECTS,
    LIFE_SCIENCES_SUBJECTS,
    HUMANITIES_SOCIAL_SUBJECTS,
    CATEGORY_COLORS_WITH_ALL,
)
from plot_subject_scatter import load_graph_df
from constants import (
    MATH_PHYSICS_LOGIC_NAME,
    LIFE_SCIENCES_NAME,
    HUMANITIES_SOCIAL_NAME,
)


def create_k_correlation_timeseries_styled(
    combined_df,
    max_k=50,
    correlation_targets=None,
    model_name=None,
    require_correct=-1,
):
    """
    Create a time series plot with styling matching the box plot (6x4 figure, consistent fonts).
    """
    # Default correlation targets if not specified
    if correlation_targets is None:
        correlation_targets = [
            "thinking_improvement"
        ]  # Internal names stay the same
    elif isinstance(correlation_targets, str):
        correlation_targets = [correlation_targets]

    merged_df = combined_df

    # Get k-column names
    k_cols = [f"total_kl_k{k}" for k in range(1, max_k + 1)]
    available_k_cols = [col for col in k_cols if col in merged_df.columns]

    if not available_k_cols:
        print("No k-profile columns found in data")
        return

    # Extract k values
    k_values = [int(col.replace("total_kl_k", "")) for col in available_k_cols]

    # Create separate plots for each correlation target
    for target_idx, target_col in enumerate(correlation_targets):
        if target_col not in merged_df.columns:
            print(f"Warning: '{target_col}' not found in dataframe columns")
            continue

        # Match box plot figure size
        fig, ax = plt.subplots(figsize=(6, 4))

        # Calculate correlations for each domain
        domain_correlations = {
            MATH_PHYSICS_LOGIC_NAME: [],
            LIFE_SCIENCES_NAME: [],
            HUMANITIES_SOCIAL_NAME: [],
        }

        domain_stderr = {
            MATH_PHYSICS_LOGIC_NAME: [],
            LIFE_SCIENCES_NAME: [],
            HUMANITIES_SOCIAL_NAME: [],
        }

        domain_counts = {
            MATH_PHYSICS_LOGIC_NAME: 0,
            LIFE_SCIENCES_NAME: 0,
            HUMANITIES_SOCIAL_NAME: 0,
        }

        # Calculate correlations for each k
        for k in k_values:
            k_col = f"total_kl_k{k}"

            # MATH_PHYSICS_LOGIC_NAME correlation
            stem_data = merged_df[
                merged_df["subject"].isin(STEM_LOGIC_SUBJECTS)
            ][[k_col, target_col]].dropna()
            if len(stem_data) > 10:
                corr, p_val = stats.spearmanr(
                    stem_data[k_col], stem_data[target_col], nan_policy="omit"
                )
                # Check if correlation is valid (not NaN)
                if not np.isnan(corr):
                    # Clamp correlation to avoid arctanh infinity
                    corr = np.clip(corr, -0.9999, 0.9999)
                    domain_correlations[MATH_PHYSICS_LOGIC_NAME].append(corr)
                    z = np.arctanh(corr)
                    se_z = 1 / np.sqrt(len(stem_data) - 3)
                    domain_stderr[MATH_PHYSICS_LOGIC_NAME].append(se_z)
                else:
                    domain_correlations[MATH_PHYSICS_LOGIC_NAME].append(np.nan)
                    domain_stderr[MATH_PHYSICS_LOGIC_NAME].append(np.nan)
                if k == 1:
                    domain_counts[MATH_PHYSICS_LOGIC_NAME] = len(stem_data)
            else:
                domain_correlations[MATH_PHYSICS_LOGIC_NAME].append(np.nan)
                domain_stderr[MATH_PHYSICS_LOGIC_NAME].append(np.nan)

            # Life Sciences correlation
            life_data = merged_df[
                merged_df["subject"].isin(LIFE_SCIENCES_SUBJECTS)
            ][[k_col, target_col]].dropna()
            if len(life_data) > 10:
                corr, p_val = stats.spearmanr(
                    life_data[k_col], life_data[target_col], nan_policy="omit"
                )
                # Check if correlation is valid (not NaN)
                if not np.isnan(corr):
                    # Clamp correlation to avoid arctanh infinity
                    corr = np.clip(corr, -0.9999, 0.9999)
                    domain_correlations["Life Sciences"].append(corr)
                    z = np.arctanh(corr)
                    se_z = 1 / np.sqrt(len(life_data) - 3)
                    domain_stderr["Life Sciences"].append(se_z)
                else:
                    domain_correlations["Life Sciences"].append(np.nan)
                    domain_stderr["Life Sciences"].append(np.nan)
                if k == 1:
                    domain_counts["Life Sciences"] = len(life_data)
            else:
                domain_correlations["Life Sciences"].append(np.nan)
                domain_stderr["Life Sciences"].append(np.nan)

            # HUMANITIES_SOCIAL_NAME correlation
            hum_data = merged_df[
                merged_df["subject"].isin(HUMANITIES_SOCIAL_SUBJECTS)
            ][[k_col, target_col]].dropna()
            if len(hum_data) > 10:
                corr, p_val = stats.spearmanr(
                    hum_data[k_col], hum_data[target_col], nan_policy="omit"
                )
                # Check if correlation is valid (not NaN)
                if not np.isnan(corr):
                    # Clamp correlation to avoid arctanh infinity
                    corr = np.clip(corr, -0.9999, 0.9999)
                    domain_correlations[HUMANITIES_SOCIAL_NAME].append(corr)
                    z = np.arctanh(corr)
                    se_z = 1 / np.sqrt(len(hum_data) - 3)
                    domain_stderr[HUMANITIES_SOCIAL_NAME].append(se_z)
                else:
                    domain_correlations[HUMANITIES_SOCIAL_NAME].append(np.nan)
                    domain_stderr[HUMANITIES_SOCIAL_NAME].append(np.nan)
                if k == 1:
                    domain_counts[HUMANITIES_SOCIAL_NAME] = len(hum_data)
            else:
                domain_correlations[HUMANITIES_SOCIAL_NAME].append(np.nan)
                domain_stderr[HUMANITIES_SOCIAL_NAME].append(np.nan)

        # Color scheme
        colors = {
            k: v for k, v in CATEGORY_COLORS_WITH_ALL.items() if k != "All"
        }

        # Plot lines for each domain
        for domain in [
            MATH_PHYSICS_LOGIC_NAME,
            LIFE_SCIENCES_NAME,
            HUMANITIES_SOCIAL_NAME,
        ]:
            correlations = np.array(domain_correlations[domain])
            stderrs = np.array(domain_stderr[domain])

            # Filter out NaN values
            valid_mask = ~np.isnan(correlations)
            if np.sum(valid_mask) > 0:
                # Ensure arrays are 1D for indexing
                valid_mask = np.asarray(valid_mask).flatten()
                valid_k = np.array(k_values)[valid_mask]
                valid_corr = correlations[valid_mask]
                valid_se = stderrs[valid_mask]

                # Plot line with error band
                label = f"{domain}"
                ax.plot(
                    valid_k,
                    valid_corr,
                    "o-",
                    color=colors[domain],
                    label=label,
                    linewidth=2,
                    markersize=4,
                    alpha=0.9,
                )

                # Add error bands (clamp correlations to avoid arctanh infinity)
                valid_corr_clamped = np.clip(valid_corr, -0.9999, 0.9999)
                upper_bound = np.tanh(np.arctanh(valid_corr_clamped) + valid_se)
                lower_bound = np.tanh(np.arctanh(valid_corr_clamped) - valid_se)
                ax.fill_between(
                    valid_k,
                    lower_bound,
                    upper_bound,
                    color=colors[domain],
                    alpha=0.15,
                )

        # Add horizontal line at y=0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=0.5)

        # Formatting to match box plot
        target_label = f'{target_col.replace("_", " ").title()}'
        # Replace "Thinking" with "Reasoning" in labels
        target_label = target_label.replace("Thinking", "Reasoning")
        ax.set_xlabel("Distance k", fontsize=14, fontweight="medium")
        ax.set_ylabel(f"Correlation", fontsize=14, fontweight="medium")
        ax.set_title(
            f"Correlation with {target_label}",
            fontsize=14,
            fontweight="bold",
        )

        # Match box plot tick settings
        # Set ticks to project outward for both x and y axes
        ax.tick_params(
            axis="x", labelsize=12, direction="out", length=4, width=1
        )
        ax.tick_params(
            axis="y", labelsize=12, direction="out", length=4, width=1
        )

        # Remove top and right spines, make left and bottom black
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["bottom"].set_color("black")

        # Set x-axis to show key k values
        if max_k <= 20:
            ax.set_xticks(range(1, max_k + 1, 2))
        elif max_k <= 50:
            ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        else:
            ax.set_xticks(range(0, max_k + 1, 10))

        # Grid
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        # Legend
        ax.legend(
            loc="best", fontsize=10, frameon=True, fancybox=False, shadow=False
        )

        plt.tight_layout()

        # Save
        os.makedirs("plots", exist_ok=True)
        output_path = f"plots/k_correlation_timeseries_{target_col}_rc{require_correct}_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show(block=False)
        print(f"Saved to {output_path}")


def create_within_subject_correlation_timeseries(
    combined_df,
    max_k=50,
    correlation_targets=None,
    model_name=None,
    require_correct=-1,
):
    """
    Compute within-subject correlations first, then average across subjects.
    This controls for between-subject differences.
    """
    if correlation_targets is None:
        correlation_targets = [
            "thinking_improvement"
        ]  # Internal names stay the same
    elif isinstance(correlation_targets, str):
        correlation_targets = [correlation_targets]

    merged_df = combined_df

    # Get k-column names
    k_cols = [f"total_kl_k{k}" for k in range(1, max_k + 1)]
    # print(combined_df["total_kl_k1"])
    # quit()
    available_k_cols = [col for col in k_cols if col in merged_df.columns]

    if not available_k_cols:
        print("No k-profile columns found in data")
        return

    k_values = [int(col.replace("total_kl_k", "")) for col in available_k_cols]

    # Get unique subjects
    subjects = merged_df["subject"].unique()

    for target_col in correlation_targets:
        if target_col not in merged_df.columns:
            print(f"Warning: '{target_col}' not found in dataframe columns")
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        # Styling to match box plot
        ax.axhline(
            y=0, color="black", linestyle="--", linewidth=1.5
        )  # alpha=0.3,

        # Store correlations by domain
        domain_subject_corrs = {
            MATH_PHYSICS_LOGIC_NAME: {
                subj: [] for subj in subjects if subj in STEM_LOGIC_SUBJECTS
            },
            "Life Sciences": {
                subj: [] for subj in subjects if subj in LIFE_SCIENCES_SUBJECTS
            },
            HUMANITIES_SOCIAL_NAME: {
                subj: []
                for subj in subjects
                if subj in HUMANITIES_SOCIAL_SUBJECTS
            },
        }

        # Compute within-subject correlations for each k
        for k in k_values:
            k_col = f"total_kl_k{k}"

            for subject in subjects:
                subj_data = merged_df[merged_df["subject"] == subject][
                    [k_col, target_col]
                ].dropna()

                if len(subj_data) >= 10:  # Need sufficient data for correlation
                    # Check if there's variance in both variables (ensure scalar values)

                    corr, p = stats.spearmanr(
                        subj_data[k_col],
                        subj_data[target_col],
                        nan_policy="omit",
                    )

                    # print(f"{subject=}")
                    # print(f"{STEM_LOGIC_SUBJECTS=}")
                    # quit()

                    # Only append if not NaN
                    if not np.isnan(corr):
                        # Assign to appropriate domain
                        if subject in STEM_LOGIC_SUBJECTS:
                            domain_subject_corrs[MATH_PHYSICS_LOGIC_NAME][
                                subject
                            ].append(corr)
                        elif subject in LIFE_SCIENCES_SUBJECTS:
                            domain_subject_corrs[LIFE_SCIENCES_NAME][
                                subject
                            ].append(corr)
                        elif subject in HUMANITIES_SOCIAL_SUBJECTS:
                            domain_subject_corrs[HUMANITIES_SOCIAL_NAME][
                                subject
                            ].append(corr)

        # Average across subjects within each domain
        colors = {
            k: v for k, v in CATEGORY_COLORS_WITH_ALL.items() if k != "All"
        }

        for domain in [
            MATH_PHYSICS_LOGIC_NAME,
            LIFE_SCIENCES_NAME,
            HUMANITIES_SOCIAL_NAME,
        ]:
            # Collect all correlations for this domain at each k
            domain_avg_corrs = []
            domain_se_corrs = []

            for k_idx in range(len(k_values)):
                k_corrs = []
                for subject, corr_list in domain_subject_corrs[domain].items():
                    if k_idx < len(corr_list):
                        k_corrs.append(corr_list[k_idx])

                if len(k_corrs) > 0:
                    # Fisher z-transform for averaging correlations
                    z_scores = []
                    for c in k_corrs:
                        # Ensure c is scalar
                        c_scalar = float(c) if not np.isnan(c) else np.nan
                        if not np.isnan(c_scalar):
                            # Clamp correlation to avoid arctanh infinity
                            c_scalar = np.clip(c_scalar, -0.9999, 0.9999)
                            z_scores.append(np.arctanh(c_scalar))
                    if z_scores:
                        avg_z = np.mean(z_scores)
                        se_z = np.std(z_scores) / np.sqrt(len(z_scores))
                        avg_corr = np.tanh(avg_z)
                        domain_avg_corrs.append(avg_corr)
                        domain_se_corrs.append(se_z)
                    else:
                        domain_avg_corrs.append(np.nan)
                        domain_se_corrs.append(np.nan)
                else:
                    domain_avg_corrs.append(np.nan)
                    domain_se_corrs.append(np.nan)

            # Plot the averaged within-subject correlations
            correlations = np.array(domain_avg_corrs)
            stderrs = np.array(domain_se_corrs)

            valid_mask = ~np.isnan(correlations)
            if np.sum(valid_mask) > 0:
                valid_k = np.array(k_values)[valid_mask]
                valid_corr = correlations[valid_mask]
                valid_se = stderrs[valid_mask]

                n_subjects = len(
                    [
                        s
                        for s in domain_subject_corrs[domain]
                        if domain_subject_corrs[domain][s]
                    ]
                )
                label = f"{domain}"

                ax.plot(
                    valid_k,
                    valid_corr,
                    "o-",
                    color=colors[domain],
                    label=label,
                    linewidth=2,
                    markersize=4,
                    alpha=0.9,
                )

                # Error bands (clamp correlations to avoid arctanh infinity)
                valid_corr_clamped = np.clip(valid_corr, -0.9999, 0.9999)
                upper_bound = np.tanh(np.arctanh(valid_corr_clamped) + valid_se)
                lower_bound = np.tanh(np.arctanh(valid_corr_clamped) - valid_se)
                ax.fill_between(
                    valid_k,
                    lower_bound,
                    upper_bound,
                    color=colors[domain],
                    alpha=0.15,
                )

        target_label = f'{target_col.replace("_", " ").title()}'
        # Replace "Thinking" with "Reasoning" in labels
        target_label = target_label.replace("Thinking", "Reasoning")
        ax.set_xlabel("Distance k", fontsize=14, fontweight="medium")
        ax.set_ylabel("Mean correlation (r)", fontsize=14, fontweight="medium")
        ax.set_title(
            f"Within-Subject Correlation with {target_label}",
            fontsize=14,
            # fontweight="bold",
        )

        # Set ticks to project outward for both x and y axes
        ax.tick_params(
            axis="x", labelsize=12, direction="out", length=4, width=1
        )
        ax.tick_params(
            axis="y", labelsize=12, direction="out", length=4, width=1
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["left"].set_color("black")
        # ax.spines["bottom"].set_linewidth(1.5)
        # ax.spines["bottom"].set_color("black")

        if max_k <= 20:
            ax.set_xticks(range(1, max_k + 1, 2))
        elif max_k <= 50:
            ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(
            loc="best", fontsize=10, frameon=True, fancybox=False, shadow=False
        )
        plt.xlim(0.5, max_k + 0.5)

        plt.tight_layout()

        os.makedirs("plots", exist_ok=True)
        output_path = f"plots/within_subject_k_correlation_{target_col}_rc{require_correct}_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show(block=False)
        print(f"Saved within-subject correlation plot to {output_path}")


if __name__ == "__main__":
    # for REQUIRE_CORRECT in [True, False]:
    REQUIRE_CORRECT = True
    MODEL_NAME = "qwen3-30b-a3b"
    df = load_graph_df(require_correct=REQUIRE_CORRECT, model=MODEL_NAME)
    # create_k_correlation_timeseries_styled(
    #     df,
    #     max_k=50,
    #     correlation_targets=["thinking_improvement"],
    #     model_name="qwen3-30b-a3b",
    #     require_correct=True,
    # )

    create_within_subject_correlation_timeseries(
        df,
        max_k=50,
        correlation_targets=["thinking_accuracy"],
        model_name=MODEL_NAME,
        require_correct=REQUIRE_CORRECT,
    )
