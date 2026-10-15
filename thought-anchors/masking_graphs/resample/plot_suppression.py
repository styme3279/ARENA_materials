#!/usr/bin/env python
"""
Plotting functions for suppression object analysis.
Separated from prep_suppobj.py for better organization.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# Add parent directory to path to import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    STEM_LOGIC_SUBJECTS,
    LIFE_SCIENCES_SUBJECTS,
    HUMANITIES_SOCIAL_SUBJECTS,
    CATEGORY_COLORS,
    CATEGORY_COLORS_WITH_ALL,
    MATH_PHYSICS_LOGIC_NAME,
    LIFE_SCIENCES_NAME,
    HUMANITIES_SOCIAL_NAME,
)


def get_df_kl(df):
    # Extract k-columns
    k_cols = [col for col in df.columns if col.startswith("total_kl_k")]
    if not k_cols:
        print("Warning: No k-columns found in graph dataframe")
        return

    # Extract k values from column names
    k_values = []
    for col in k_cols:
        try:
            k_val = int(col.replace("total_kl_k", ""))
            k_values.append(k_val)
        except:
            continue
    k_values = sorted(k_values)

    print(
        f"Found {len(k_cols)} k-columns, k ranging from {min(k_values)} to {max(k_values)}"
    )

    # Merge subject info with graph data
    return df[k_cols], k_cols, k_values


def plot_k_profiles_by_domain(df, model_name, require_correct=-1):
    """
    Visualize k-profiles (diagonal strengths) by subject/domain.
    Creates multiple visualizations to show how different domains utilize
    short vs long-range connections.
    """
    print("\n" + "=" * 60)
    print("GENERATING K-PROFILE VISUALIZATIONS BY DOMAIN")
    print("=" * 60)

    # Check if subject column exists
    if "subject" not in df.columns:
        print("Warning: 'subject' column not found in dataframe")
        return

    df_kl_k, k_cols, k_values = get_df_kl(df)
    df_kl_k = pd.concat([df[["subject"]], df_kl_k], axis=1)

    # Group by subject and compute mean k-profiles, also get counts
    subject_profiles = df_kl_k.groupby("subject")[k_cols].mean()
    subject_counts = df_kl_k.groupby("subject").size().to_dict()

    # Use imported subject category constants
    stem_logic_subjects = STEM_LOGIC_SUBJECTS
    life_sciences_subjects = LIFE_SCIENCES_SUBJECTS
    humanities_social_subjects = HUMANITIES_SOCIAL_SUBJECTS

    create_k_profile_summary(
        subject_profiles,
        k_values,
        stem_logic_subjects,
        life_sciences_subjects,
        humanities_social_subjects,
        subject_counts,
        model_name,
        require_correct,
    )
    create_decay_rate_comparison(
        subject_profiles,
        k_values,
        stem_logic_subjects,
        life_sciences_subjects,
        humanities_social_subjects,
        subject_counts,
        model_name,
        require_correct,
    )


def create_k_profile_heatmap(
    subject_profiles,
    k_values,
    stem_logic_subjects,
    life_sciences_subjects,
    humanities_social_subjects,
    subject_counts,
    require_correct=-1,
    model_name=None,
):
    """Create heatmap showing k-profiles for all subjects."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Calculate vmax and vmin for each individual subject
    subject_vmaxes = []
    subject_vmins = []

    for subject in subject_profiles.index:
        subject_data = subject_profiles.loc[subject].values
        subject_data = subject_data[~np.isnan(subject_data)]
        if len(subject_data) > 0:
            # Use actual max for vmax, actual min for vmin for each subject
            subject_vmaxes.append(np.max(subject_data))
            subject_vmins.append(np.min(subject_data))

    # Use the highest vmax from any subject and lowest vmin from any subject
    if subject_vmaxes:
        vmax = max(subject_vmaxes)
        vmin = min(subject_vmins)
        # Find which subject has the max vmax
        max_subject_idx = np.argmax(subject_vmaxes)
        max_subject = subject_profiles.index[max_subject_idx]
        min_subject_idx = np.argmin(subject_vmins)
        min_subject = subject_profiles.index[min_subject_idx]
        print(
            f"Using color scale: vmin={vmin:.3f} (from {min_subject}), vmax={vmax:.3f} (from {max_subject})"
        )
    else:
        vmin, vmax = 0, 1  # Default range
        print(f"Using default color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")

    # Helper function to create heatmap for category
    def plot_category_heatmap(ax, subjects, category_name, cmap="coolwarm"):
        # Filter subjects that exist in the data
        available_subjects = [
            s for s in subjects if s in subject_profiles.index
        ]
        if not available_subjects:
            ax.text(
                0.5,
                0.5,
                f"No {category_name} subjects found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return None

        # Get data for these subjects
        category_data = subject_profiles.loc[available_subjects]

        # Convert to numpy array for plotting
        data_array = category_data.values

        # Create heatmap with consistent vmin/vmax
        im = ax.imshow(
            data_array,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        # Set labels with counts
        ax.set_yticks(range(len(available_subjects)))
        labels = []
        for s in available_subjects:
            label = s.replace("_", " ").title()[:25]
            count = subject_counts.get(s, 0)
            labels.append(f"{label} (N={count})")
        ax.set_yticklabels(labels, fontsize=9)

        # Set x-ticks at regular intervals
        k_tick_indices = list(
            range(0, len(k_values), max(1, len(k_values) // 20))
        )
        ax.set_xticks(k_tick_indices)
        ax.set_xticklabels([k_values[i] for i in k_tick_indices])

        ax.set_xlabel("Distance (k)", fontsize=10)
        ax.set_title(
            f"{category_name} Subjects - Connection Strength by Distance",
            fontsize=12,
            fontweight="bold",
        )

        return im  # Return the image for colorbar

    # Plot each category (subject_counts is accessible from parent scope)
    im1 = plot_category_heatmap(axes[0], stem_logic_subjects, MATH_PHYSICS_LOGIC_NAME)
    im2 = plot_category_heatmap(
        axes[1], life_sciences_subjects, "Life Sciences"
    )
    im3 = plot_category_heatmap(
        axes[2], humanities_social_subjects, HUMANITIES_SOCIAL_NAME
    )

    # Add a single colorbar for all subplots (since they share the same scale)
    if im1 is not None or im2 is not None or im3 is not None:
        # Use the first non-None image for the colorbar
        im_for_colorbar = (
            im1 if im1 is not None else (im2 if im2 is not None else im3)
        )
        cbar = fig.colorbar(
            im_for_colorbar,
            ax=axes,
            orientation="vertical",
            fraction=0.02,
            pad=0.02,
        )
        cbar.set_label("Connection Strength", rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(
        f"results/graph_data/{model_name}/k_profiles_heatmap_by_domain_rc{require_correct}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show(block=False)
    print(
        f"Saved heatmap to results/k_profiles_heatmap_by_domain_rc{require_correct}.png"
    )


def create_k_profile_summary(
    subject_profiles,
    k_values,
    stem_logic_subjects,
    life_sciences_subjects,
    humanities_social_subjects,
    subject_counts,
    model_name,
    require_correct=-1,
):
    """Create summary plot showing averaged profiles by domain category."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define k-ranges for summary statistics
    local_range = range(1, 6)  # k=1 to 5
    medium_range = range(6, 21)  # k=6 to 20
    long_range = range(21, min(51, max(k_values) + 1))  # k=21 to 50

    # Compute average profiles for each category
    categories = {
        MATH_PHYSICS_LOGIC_NAME: stem_logic_subjects,
        "Life Sciences": life_sciences_subjects,
        HUMANITIES_SOCIAL_NAME: humanities_social_subjects,
    }

    colors = CATEGORY_COLORS

    for category_name, subjects in categories.items():
        available_subjects = [
            s for s in subjects if s in subject_profiles.index
        ]
        if not available_subjects:
            continue

        # Calculate total N for this category
        total_n = sum(subject_counts.get(s, 0) for s in available_subjects)
        num_subjects = len(available_subjects)

        # Get mean profile for this category
        category_data = subject_profiles.loc[available_subjects].mean(axis=0)

        # Plot with error bars (std)
        category_std = subject_profiles.loc[available_subjects].std(axis=0)

        # Include counts in label
        label = f"{category_name} ({num_subjects} subjects, N={total_n})"
        ax.plot(
            k_values,
            category_data.values,
            label=label,
            color=colors[category_name],
            linewidth=2,
            alpha=0.8,
        )
        ax.fill_between(
            k_values,
            category_data.values - category_std.values,
            category_data.values + category_std.values,
            color=colors[category_name],
            alpha=0.2,
        )

    # Add vertical lines to separate k-ranges
    ax.axvline(
        x=5,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Local/Medium boundary",
    )
    ax.axvline(
        x=20,
        color="gray",
        linestyle="-.",
        alpha=0.5,
        label="Medium/Long boundary",
    )

    ax.set_xlabel("Distance (k)", fontsize=12)
    ax.set_ylabel("Mean Connection Strength", fontsize=12)
    ax.set_title(
        "Connection Strength Decay by Domain Category",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Use log scale for x-axis to better show the decay
    ax.set_xscale("log")
    ax.set_xlim(1, max(k_values))

    plt.tight_layout()
    model_dir = f"results/graph_data/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    plt.savefig(
        f"{model_dir}/k_profiles_summary_by_domain_rc{require_correct}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show(block=False)
    print(
        f"Saved summary plot to {model_dir}/k_profiles_summary_by_domain_rc{require_correct}.png"
    )


def create_decay_rate_comparison(
    subject_profiles,
    k_values,
    stem_logic_subjects,
    life_sciences_subjects,
    humanities_social_subjects,
    subject_counts,
    model_name,
    require_correct=-1,
):
    """Create bar plot comparing local vs long-range strength by domain."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Compute summary statistics for each subject
    summary_stats = []

    for subject in subject_profiles.index:
        profile = subject_profiles.loc[subject]

        # Compute local strength (k=1-5)
        local_cols = [
            f"total_kl_k{k}"
            for k in range(1, 6)
            if f"total_kl_k{k}" in profile.index
        ]
        if local_cols:
            local_values = profile[local_cols].values
            local_values = local_values[
                ~(np.isnan(local_values) | np.isinf(local_values))
            ]
            local_strength = (
                float(np.mean(local_values)) if len(local_values) > 0 else 0
            )
        else:
            local_strength = 0

        # Compute long-range strength (k=21-50)
        long_cols = [
            f"total_kl_k{k}"
            for k in range(21, 51)
            if f"total_kl_k{k}" in profile.index
        ]
        if long_cols:
            long_values = profile[long_cols].values
            long_values = long_values[
                ~(np.isnan(long_values) | np.isinf(long_values))
            ]
            long_strength = (
                float(np.mean(long_values)) if len(long_values) > 0 else 0
            )
        else:
            long_strength = 0

        # Compute decay rate (slope from k=1 to k=20)
        early_cols = [
            f"total_kl_k{k}"
            for k in range(1, 21)
            if f"total_kl_k{k}" in profile.index
        ]
        if len(early_cols) > 5:
            try:
                x = np.array(
                    [int(col.replace("total_kl_k", "")) for col in early_cols]
                )
                y = profile[early_cols].values

                # Check for NaN or infinite values
                valid_mask = ~(np.isnan(y) | np.isinf(y))
                if (
                    np.sum(valid_mask) < 3
                ):  # Need at least 3 points for meaningful fit
                    decay_rate = 0
                else:
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]

                    # Use try-except for polyfit in case of numerical issues
                    try:
                        # Fit log-linear model
                        slope, _ = np.polyfit(np.log(x_valid), y_valid, 1)
                        decay_rate = -slope  # Negative slope means decay
                    except (np.linalg.LinAlgError, ValueError):
                        # If polyfit fails, try simple linear regression as fallback
                        try:
                            slope, _ = np.polyfit(x_valid, y_valid, 1)
                            decay_rate = -slope
                        except:
                            decay_rate = 0
            except Exception as e:
                print(
                    f"Warning: Could not compute decay rate for {subject}: {e}"
                )
                decay_rate = 0
        else:
            decay_rate = 0

        # Determine category
        if subject in stem_logic_subjects:
            category = MATH_PHYSICS_LOGIC_NAME
        elif subject in life_sciences_subjects:
            category = "Life Sciences"
        elif subject in humanities_social_subjects:
            category = HUMANITIES_SOCIAL_NAME
        else:
            category = "Other"

        summary_stats.append(
            {
                "subject": subject,
                "category": category,
                "local_strength": local_strength,
                "long_strength": long_strength,
                "local_to_long_ratio": local_strength / (long_strength + 1e-10),
                "decay_rate": decay_rate,
            }
        )

    summary_df = pd.DataFrame(summary_stats)

    # Plot 1: Local vs Long-range strength
    ax1 = axes[0]
    categories = [MATH_PHYSICS_LOGIC_NAME, LIFE_SCIENCES_NAME, HUMANITIES_SOCIAL_NAME]
    colors = ["#FF6B6B", "#95E77E", "#4ECDC4"]

    x_pos = np.arange(len(categories))
    width = 0.35

    # Calculate means, standard errors, and totals for labels
    category_stats = {}
    for cat in categories:
        cat_df = summary_df[summary_df["category"] == cat]
        cat_subjects = cat_df["subject"].tolist()
        total_n = sum(subject_counts.get(s, 0) for s in cat_subjects)
        n = len(cat_df)

        if n > 0:
            local_mean = cat_df["local_strength"].mean()
            long_mean = cat_df["long_strength"].mean()
            local_std = cat_df["local_strength"].std()
            long_std = cat_df["long_strength"].std()
            # Calculate standard errors
            local_se = local_std / np.sqrt(n) if n > 0 else 0
            long_se = long_std / np.sqrt(n) if n > 0 else 0
            # Handle NaN values
            local_mean = 0 if np.isnan(local_mean) else local_mean
            long_mean = 0 if np.isnan(long_mean) else long_mean
            local_se = 0 if np.isnan(local_se) else local_se
            long_se = 0 if np.isnan(long_se) else long_se
        else:
            local_mean = long_mean = local_se = long_se = 0

        category_stats[cat] = {
            "local_mean": local_mean,
            "long_mean": long_mean,
            "local_se": local_se,
            "long_se": long_se,
            "n_subjects": n,
            "total_n": total_n,
        }

    local_means = [category_stats[cat]["local_mean"] for cat in categories]
    long_means = [category_stats[cat]["long_mean"] for cat in categories]
    local_ses = [category_stats[cat]["local_se"] for cat in categories]
    long_ses = [category_stats[cat]["long_se"] for cat in categories]

    bars1 = ax1.bar(
        x_pos - width / 2,
        local_means,
        width,
        label="Local (k=1-5)",
        color="skyblue",
        yerr=local_ses,
        capsize=5,
    )
    bars2 = ax1.bar(
        x_pos + width / 2,
        long_means,
        width,
        label="Long-range (k=21-50)",
        color="coral",
        yerr=long_ses,
        capsize=5,
    )

    # Create x-labels with counts
    x_labels = [
        f"{cat}\n({category_stats[cat]['n_subjects']} subj, N={category_stats[cat]['total_n']})"
        for cat in categories
    ]

    # Set y-axis limits to better show differences (not starting at 0)
    all_values = local_means + long_means
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        padding = (max_val - min_val) * 0.1
        ax1.set_ylim(
            max(0, min_val - padding), max_val + padding
        )  # Don't go below 0 for bar charts

    ax1.set_xlabel("Domain Category", fontsize=12)
    ax1.set_ylabel("Mean Connection Strength", fontsize=12)
    ax1.set_title(
        "Local vs Long-range Connection Strength",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Decay rate comparison
    ax2 = axes[1]
    decay_means = []
    decay_ses = []  # Standard errors instead of standard deviations
    for cat in categories:
        cat_df = summary_df[summary_df["category"] == cat]
        if len(cat_df) > 0:
            decay_mean = cat_df["decay_rate"].mean()
            decay_std = cat_df["decay_rate"].std()
            n = len(cat_df)
            # Calculate standard error = std / sqrt(n)
            decay_se = decay_std / np.sqrt(n) if n > 0 else 0
            # Handle NaN values
            decay_means.append(0 if np.isnan(decay_mean) else decay_mean)
            decay_ses.append(0 if np.isnan(decay_se) else decay_se)
        else:
            decay_means.append(0)
            decay_ses.append(0)

    bars = ax2.bar(x_pos, decay_means, color=colors, alpha=0.7)
    ax2.errorbar(
        x_pos, decay_means, yerr=decay_ses, fmt="none", color="black", capsize=5
    )

    # Set y-axis limits to better show differences (not starting at 0)
    if decay_means:
        # Calculate range with some padding
        min_val = (
            min(decay_means) - max(decay_ses) if decay_ses else min(decay_means)
        )
        max_val = (
            max(decay_means) + max(decay_ses) if decay_ses else max(decay_means)
        )
        padding = (max_val - min_val) * 0.1
        ax2.set_ylim(min_val - padding, max_val + padding)

    ax2.set_xlabel("Domain Category", fontsize=12)
    ax2.set_ylabel("Decay Rate (higher = faster decay)", fontsize=12)
    ax2.set_title(
        "Connection Strength Decay Rate", fontsize=13, fontweight="bold"
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)  # Use the same labels with counts
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    model_dir = f"results/graph_data/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    plt.savefig(
        f"{model_dir}/k_decay_comparison_by_domain_rc{require_correct}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show(block=False)
    print(
        f"Saved decay comparison to {model_dir}/k_decay_comparison_by_domain_rc{require_correct}.png"
    )

    # Print summary statistics
    print("\nSummary Statistics by Category:")
    print("-" * 60)
    for cat in categories:
        cat_df = summary_df[summary_df["category"] == cat]
        if len(cat_df) > 0:
            cat_subjects = cat_df["subject"].tolist()
            total_n = sum(subject_counts.get(s, 0) for s in cat_subjects)

            print(
                f"\n{cat} ({len(cat_df)} subjects, N={total_n} total examples):"
            )
            print(
                f"  Local strength: {cat_df['local_strength'].mean():.4f} ± {cat_df['local_strength'].std():.4f}"
            )
            print(
                f"  Long strength:  {cat_df['long_strength'].mean():.4f} ± {cat_df['long_strength'].std():.4f}"
            )
            print(
                f"  Local/Long ratio: {cat_df['local_to_long_ratio'].mean():.2f}"
            )
            print(f"  Decay rate: {cat_df['decay_rate'].mean():.4f}")

            # Get top subjects with their counts
            top_local = cat_df.nlargest(1, "local_strength")["subject"].values[
                0
            ]
            top_local_n = subject_counts.get(top_local, 0)
            top_long = cat_df.nlargest(1, "long_strength")["subject"].values[0]
            top_long_n = subject_counts.get(top_long, 0)

            print(f"  Top subject (local): {top_local} (N={top_local_n})")
            print(f"  Top subject (long): {top_long} (N={top_long_n})")


def create_k_correlation_timeseries(
    combined_df,
    max_k=50,
    correlation_targets=None,
    model_name=None,
    require_correct=-1,
):
    """
    Create a time series plot showing correlation coefficients for different k values,
    with separate lines for different domains.

    Args:
        combined_df: DataFrame with thinking/non-thinking results
        graph_df: DataFrame with graph metrics including k-profile data
        max_k: Maximum k value to plot
        correlation_targets: List of columns to correlate with (default: ['thinking_improvement'])
                           Can include: 'thinking_improvement', 'thinking_is_correct',  # Internal names stay the same
                           'prob_correct', or any column in combined_df
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # Default correlation targets if not specified
    if correlation_targets is None:
        correlation_targets = ["thinking_improvement"]
    elif isinstance(correlation_targets, str):
        correlation_targets = [correlation_targets]

    # Use imported subject category constants
    stem_logic_subjects = STEM_LOGIC_SUBJECTS
    life_sciences_subjects = LIFE_SCIENCES_SUBJECTS
    humanities_social_subjects = HUMANITIES_SOCIAL_SUBJECTS

    # Since dataframes are already aligned by index, just use the combined_df which has both
    merged_df = combined_df

    # Get k-column names
    k_cols = [f"total_kl_k{k}" for k in range(1, max_k + 1)]
    available_k_cols = [col for col in k_cols if col in merged_df.columns]

    if not available_k_cols:
        print("No k-profile columns found in data")
        return

    # Extract k values
    k_values = [int(col.replace("total_kl_k", "")) for col in available_k_cols]

    # Calculate correlations for each domain and k value
    domain_correlations = {
        MATH_PHYSICS_LOGIC_NAME: [],
        "Life Sciences": [],
        HUMANITIES_SOCIAL_NAME: [],
        "All": [],
    }

    domain_stderr = {
        MATH_PHYSICS_LOGIC_NAME: [],
        "Life Sciences": [],
        HUMANITIES_SOCIAL_NAME: [],
        "All": [],
    }

    domain_counts = {
        MATH_PHYSICS_LOGIC_NAME: 0,
        "Life Sciences": 0,
        HUMANITIES_SOCIAL_NAME: 0,
        "All": len(merged_df),
    }

    # Create separate plots for each correlation target
    for target_idx, target_col in enumerate(correlation_targets):
        # Check if target column exists
        if target_col not in merged_df.columns:
            print(f"Warning: '{target_col}' not found in dataframe columns")
            continue

        # Create a new figure for each target
        fig, ax = plt.subplots(figsize=(12, 7))

        # Reset correlation dictionaries for this target
        domain_correlations = {
            MATH_PHYSICS_LOGIC_NAME: [],
            "Life Sciences": [],
            HUMANITIES_SOCIAL_NAME: [],
            "All": [],
        }

        domain_stderr = {
            MATH_PHYSICS_LOGIC_NAME: [],
            "Life Sciences": [],
            HUMANITIES_SOCIAL_NAME: [],
            "All": [],
        }

        # Calculate correlations for each k
        for k in k_values:
            k_col = f"total_kl_k{k}"

            # All subjects correlation
            valid_data = merged_df[[k_col, target_col]].dropna()
            if len(valid_data) > 10:
                corr, p_val = stats.spearmanr(
                    valid_data[k_col], valid_data[target_col], nan_policy="omit"
                )
                domain_correlations["All"].append(corr)
                # Calculate standard error using Fisher z-transformation
                z = np.arctanh(corr)
                se_z = 1 / np.sqrt(len(valid_data) - 3)
                domain_stderr["All"].append(se_z)
            else:
                domain_correlations["All"].append(np.nan)
                domain_stderr["All"].append(np.nan)

            # Mathematics, Physics, and Logic correlation
            stem_data = merged_df[
                merged_df["subject"].isin(stem_logic_subjects)
            ][[k_col, target_col]].dropna()
            if len(stem_data) > 10:
                corr, p_val = stats.spearmanr(
                    stem_data[k_col], stem_data[target_col], nan_policy="omit"
                )
                domain_correlations[MATH_PHYSICS_LOGIC_NAME].append(corr)
                z = np.arctanh(corr)
                se_z = 1 / np.sqrt(len(stem_data) - 3)
                domain_stderr[MATH_PHYSICS_LOGIC_NAME].append(se_z)
                if k == 1 and target_idx == 0:  # Count once
                    domain_counts[MATH_PHYSICS_LOGIC_NAME] = len(stem_data)
            else:
                domain_correlations[MATH_PHYSICS_LOGIC_NAME].append(np.nan)
                domain_stderr[MATH_PHYSICS_LOGIC_NAME].append(np.nan)

            # Life Sciences correlation
            life_data = merged_df[
                merged_df["subject"].isin(life_sciences_subjects)
            ][[k_col, target_col]].dropna()
            if len(life_data) > 10:
                corr, p_val = stats.spearmanr(
                    life_data[k_col], life_data[target_col], nan_policy="omit"
                )
                domain_correlations["Life Sciences"].append(corr)
                z = np.arctanh(corr)
                se_z = 1 / np.sqrt(len(life_data) - 3)
                domain_stderr["Life Sciences"].append(se_z)
                if k == 1 and target_idx == 0:  # Count once
                    domain_counts["Life Sciences"] = len(life_data)
            else:
                domain_correlations["Life Sciences"].append(np.nan)
                domain_stderr["Life Sciences"].append(np.nan)

            # Humanities/Social correlation  # Just a comment, name doesn't need constant
            hum_data = merged_df[
                merged_df["subject"].isin(humanities_social_subjects)
            ][[k_col, target_col]].dropna()
            if len(hum_data) > 10:
                corr, p_val = stats.spearmanr(
                    hum_data[k_col], hum_data[target_col], nan_policy="omit"
                )
                domain_correlations[HUMANITIES_SOCIAL_NAME].append(corr)
                z = np.arctanh(corr)
                se_z = 1 / np.sqrt(len(hum_data) - 3)
                domain_stderr[HUMANITIES_SOCIAL_NAME].append(se_z)
                if k == 1 and target_idx == 0:  # Count once
                    domain_counts[HUMANITIES_SOCIAL_NAME] = len(hum_data)
            else:
                domain_correlations[HUMANITIES_SOCIAL_NAME].append(np.nan)
                domain_stderr[HUMANITIES_SOCIAL_NAME].append(np.nan)

        # Color scheme
        colors = CATEGORY_COLORS_WITH_ALL

        # Plot lines for each domain (excluding 'All')
        for domain in [MATH_PHYSICS_LOGIC_NAME, LIFE_SCIENCES_NAME, HUMANITIES_SOCIAL_NAME]:
            correlations = np.array(domain_correlations[domain])
            stderrs = np.array(domain_stderr[domain])

            # Filter out NaN values
            valid_mask = ~np.isnan(correlations)
            if np.sum(valid_mask) > 0:
                valid_k = np.array(k_values)[valid_mask]
                valid_corr = correlations[valid_mask]
                valid_se = stderrs[valid_mask]

                # Plot line with error band
                label = f"{domain} (N={domain_counts[domain]})"
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

                # Add error bands (convert from z-scale SE back to correlation scale)
                upper_bound = np.tanh(np.arctanh(valid_corr) + valid_se)
                lower_bound = np.tanh(np.arctanh(valid_corr) - valid_se)
                ax.fill_between(
                    valid_k,
                    lower_bound,
                    upper_bound,
                    color=colors[domain],
                    alpha=0.15,
                )

        # Add horizontal line at y=0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=0.5)

        # Formatting - add brackets around the dependent variable
        target_label = f'[{target_col.replace("_", " ").title()}]'
        ax.set_xlabel("Distance k (diagonal offset)", fontsize=12)
        ax.set_ylabel(f"Correlation with {target_label}", fontsize=12)
        ax.set_title(
            f"Correlation Between Causal Connection Strength and {target_label}\nby Distance and Domain",
            fontsize=14,
            fontweight="bold",
        )

        # Set x-axis to show key k values
        if max_k <= 20:
            ax.set_xticks(range(1, max_k + 1, 2))
        elif max_k <= 50:
            ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        else:
            ax.set_xticks(range(0, max_k + 1, 10))

        # Grid
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.15, which="minor", linestyle=":", linewidth=0.5)

        # Legend
        ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)

        # Print summary statistics for this target
        print(f"\nCorrelation Summary for {target_label}:")
        print("-" * 60)
        for domain in [MATH_PHYSICS_LOGIC_NAME, LIFE_SCIENCES_NAME, HUMANITIES_SOCIAL_NAME]:
            if domain not in domain_correlations:
                continue
            correlations = np.array(domain_correlations[domain])
            valid_corr = correlations[~np.isnan(correlations)]
            if len(valid_corr) > 0:
                print(f"\n{domain} (N={domain_counts[domain]}):")
                print(f"  Mean correlation: {np.mean(valid_corr):.4f}")
                print(
                    f"  Max correlation: {np.max(valid_corr):.4f} at k={k_values[np.nanargmax(correlations)]}"
                )
                print(
                    f"  Min correlation: {np.min(valid_corr):.4f} at k={k_values[np.nanargmin(correlations)]}"
                )

                # Find crossover point (where correlation becomes negative)
                neg_indices = np.where(valid_corr < 0)[0]
                if len(neg_indices) > 0:
                    first_neg_idx = neg_indices[0]
                    crossover_k = np.array(k_values)[~np.isnan(correlations)][
                        first_neg_idx
                    ]
                    print(f"  Crossover to negative: k={crossover_k}")
                else:
                    print(f"  Crossover to negative: Never")

        # Adjust layout and save for this target
        plt.tight_layout()
        # Save with filename based on this specific target
        model_dir = f"results/{model_name.replace('/', '-')}"
        os.makedirs(model_dir, exist_ok=True)
        plt.savefig(
            f"{model_dir}/k_correlation_timeseries_{target_col}_rc{require_correct}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show(block=False)

        print(
            f"\nSaved correlation time series to {model_dir}/k_correlation_timeseries_{target_col}_rc{require_correct}.png"
        )


def create_accuracy_vs_kl_scatter(df, model_name, require_correct=-1):
    """
    Create scatter plots and interactive HTML showing performance metrics vs KL effects.
    Creates separate plots for:
    - Non-thinking accuracy vs KL
    - Reasoning accuracy vs KL
    - Reasoning improvement vs KL
    For both local (k=1-5) and long-range (k=20-50) KL averages.

    Args:
        combined_df: DataFrame with subject and accuracy information
        graph_df: DataFrame with KL divergence values for different k
    """
    print("\n" + "=" * 60)
    print("GENERATING ACCURACY VS KL SCATTER PLOTS")
    print("=" * 60)

    # Check if subject column exists
    if "subject" not in df.columns:
        print("Warning: 'subject' column not found in dataframe")
        return

    df_kl_k, k_cols, k_values = get_df_kl(df)
    df_kl_k = pd.concat([df, df_kl_k], axis=1)

    # Calculate local KL average (k=1-5)
    local_k_cols = [
        f"total_kl_k{k}" for k in range(1, 6) if f"total_kl_k{k}" in k_cols
    ]
    if local_k_cols:
        df_kl_k["local_kl_avg"] = df_kl_k[local_k_cols].mean(axis=1)
    else:
        print("Warning: No local k columns (k=1-5) found")
        return

    # Calculate long-range KL average (k=20-50)
    long_k_cols = [
        f"total_kl_k{k}" for k in range(20, 51) if f"total_kl_k{k}" in k_cols
    ]
    if long_k_cols:
        df_kl_k["long_kl_avg"] = df_kl_k[long_k_cols].mean(axis=1)
    else:
        print("Warning: No long-range k columns (k=20-50) found")
        return

    # Calculate thinking improvement if not already present
    if "thinking_improvement" not in df_kl_k.columns:
        # Use thinking_is_correct as a float for accuracy
        df_kl_k["thinking_accuracy"] = df_kl_k["thinking_is_correct"].astype(  # Internal name stays the same
            float
        )
        df_kl_k["nonthinking_accuracy"] = df_kl_k["is_correct"].astype(float)
        df_kl_k["thinking_improvement"] = (
            df_kl_k["thinking_accuracy"] - df_kl_k["prob_correct"]
        )
    else:
        df_kl_k["thinking_accuracy"] = df_kl_k["thinking_is_correct"].astype(  # Internal name stays the same
            float
        )
        df_kl_k["nonthinking_accuracy"] = df_kl_k["is_correct"].astype(float)

    # Calculate subject-level statistics for all metrics
    subject_stats = (
        df_kl_k.groupby("subject")
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

    # Get subject counts for sizing dots
    subject_counts = df_kl_k.groupby("subject").size().to_dict()
    subject_stats["count"] = subject_stats["subject"].map(subject_counts)

    # Print summary of subjects found
    print(f"\nFound {len(subject_stats)} subjects with data:")
    print(f"Total samples: {len(df_kl_k)}")
    print(
        f"Subjects: {', '.join(sorted(subject_stats['subject'].unique()[:10]))}{'...' if len(subject_stats) > 10 else ''}"
    )

    # Create plots for each metric combination
    metrics = ["nonthinking_accuracy", "thinking_accuracy"]
    metric_labels = ["Non-thinking Accuracy", "Reasoning Accuracy"]

    for metric, metric_label in zip(metrics, metric_labels):
        for kl_type in ["local", "long"]:
            # Create static PNG plots
            create_accuracy_kl_scatter_static(
                subject_stats,
                kl_type,
                metric,
                metric_label,
                model_name,
                require_correct,
            )

            # Create interactive HTML plots
            try:
                create_accuracy_kl_scatter_interactive(
                    subject_stats,
                    kl_type,
                    metric,
                    metric_label,
                    model_name,
                    require_correct,
                )
            except ImportError:
                print(
                    f"Warning: plotly not available, skipping interactive HTML for {metric} vs {kl_type} KL"
                )


def create_accuracy_kl_scatter_static(
    subject_stats,
    kl_type="local",
    metric="nonthinking_accuracy",
    metric_label="Non-thinking Accuracy",
    model_name=None,
    require_correct=-1,
):
    """
    Create static PNG scatter plot of performance metric vs KL effects.

    Args:
        subject_stats: DataFrame with subject-level statistics
        kl_type: "local" for k=1-5 or "long" for k=20-50
        metric: Column name for x-axis metric
        metric_label: Display label for x-axis metric
    """
    import matplotlib.pyplot as plt

    kl_col = f"{kl_type}_kl_avg"
    kl_label = (
        "Local KL Average (k=1-5)"
        if kl_type == "local"
        else "Long-range KL Average (k=20-50)"
    )

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

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
            colors.append(category_colors["STEM/Logic"])
            categories.append("STEM/Logic")
        elif subject in life_sciences_subjects:
            colors.append(category_colors["Life Sciences"])
            categories.append("Life Sciences")
        else:
            colors.append(category_colors[HUMANITIES_SOCIAL_NAME])
            categories.append(HUMANITIES_SOCIAL_NAME)

    subject_stats["category"] = categories

    # Create scatter plot by category (for legend)
    for category, color in category_colors.items():
        cat_data = subject_stats[subject_stats["category"] == category]
        if len(cat_data) > 0:
            ax.scatter(
                cat_data[metric],
                cat_data[kl_col],
                s=cat_data["count"] * 0.5 + 50,  # Size based on count
                c=color,
                label=f"{category} (n={len(cat_data)})",
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5,
            )

    # Add subject labels
    for idx, row in subject_stats.iterrows():
        # Format subject name
        subject_display = row["subject"].replace("_", " ").replace(" ", "\n", 1)

        # Position text slightly above the point
        ax.annotate(
            subject_display,
            xy=(row[metric], row[kl_col]),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=6,
            ha="center",
            va="bottom",
            alpha=0.7,
        )

    # Calculate Spearman correlation and add trend line
    from scipy import stats

    # Calculate Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(
        subject_stats[metric], subject_stats[kl_col], nan_policy="omit"
    )

    # Drop NaN values for linear regression to match spearmanr behavior
    valid_mask = ~(subject_stats[metric].isna() | subject_stats[kl_col].isna())
    valid_x = subject_stats[metric][valid_mask]
    valid_y = subject_stats[kl_col][valid_mask]

    # Still plot linear trend line for visualization
    if len(valid_x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_x, valid_y
        )
    else:
        # If not enough valid data, use zeros
        slope, intercept, r_value, p_value, std_err = 0, 0, 0, 1, 0
    x_trend = np.linspace(
        subject_stats[metric].min(), subject_stats[metric].max(), 100
    )
    y_trend = slope * x_trend + intercept
    ax.plot(
        x_trend,
        y_trend,
        "k--",
        alpha=0.5,
        linewidth=2,
        label=f"Trend (Spearman r={spearman_r:.3f}, p={spearman_p:.3f})",
    )

    # Formatting
    ax.set_xlabel(metric_label, fontsize=14)
    ax.set_ylabel(kl_label, fontsize=14)
    ax.set_title(
        f"Subject Performance: {metric_label} vs {kl_label}",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Set axis limits with padding
    x_padding = (
        subject_stats[metric].max() - subject_stats[metric].min()
    ) * 0.05
    y_padding = (
        subject_stats[kl_col].max() - subject_stats[kl_col].min()
    ) * 0.05
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
    model_dir = f"results/graph_data/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    output_path = f"{model_dir}/{metric_short}_vs_{kl_type}_kl_scatter_rc{require_correct}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    print(f"Saved {metric_label} vs {kl_type} KL scatter plot to {output_path}")


def create_accuracy_kl_scatter_interactive(
    subject_stats,
    kl_type="local",
    metric="nonthinking_accuracy",
    metric_label="Non-thinking Accuracy",
    model_name=None,
    require_correct=-1,
):
    """
    Create interactive HTML scatter plot of performance metric vs KL effects.

    Args:
        subject_stats: DataFrame with subject-level statistics
        kl_type: "local" for k=1-5 or "long" for k=20-50
        metric: Column name for x-axis metric
        metric_label: Display label for x-axis metric
    """
    try:
        import plotly.graph_objects as go
        from scipy import stats as scipy_stats
    except ImportError:
        print("Warning: plotly not available, skipping interactive plot")
        return

    kl_col = f"{kl_type}_kl_avg"
    kl_label = (
        "Local KL Average (k=1-5)"
        if kl_type == "local"
        else "Long-range KL Average (k=20-50)"
    )

    # Use imported subject category constants
    stem_logic_subjects = STEM_LOGIC_SUBJECTS
    life_sciences_subjects = LIFE_SCIENCES_SUBJECTS

    # Assign categories based on subject
    categories = []
    for subject in subject_stats["subject"]:
        if subject in stem_logic_subjects:
            categories.append(MATH_PHYSICS_LOGIC_NAME)
        elif subject in life_sciences_subjects:
            categories.append(LIFE_SCIENCES_NAME)
        else:
            categories.append(HUMANITIES_SOCIAL_NAME)

    subject_stats["category"] = categories

    # Define color mapping for categories
    color_map = {
        "STEM/Logic": "#FF6B6B",
        "Life Sciences": "#95E77E",
        HUMANITIES_SOCIAL_NAME: "#4ECDC4",
    }

    # Create figure
    fig = go.Figure()

    # Add scatter points for each category
    for category in [MATH_PHYSICS_LOGIC_NAME, LIFE_SCIENCES_NAME, HUMANITIES_SOCIAL_NAME]:
        df_cat = subject_stats[subject_stats["category"] == category]
        if len(df_cat) == 0:
            continue

        # Create hover text
        hover_text = []
        for idx, row in df_cat.iterrows():
            if metric == "thinking_improvement":
                metric_display = f"{metric_label}: {row[metric]:.1%}"
            else:
                metric_display = f"{metric_label}: {row[metric]:.1%}"

            text = (
                f"<b>{row['subject'].replace('_', ' ').title()}</b><br>"
                f"{metric_display}<br>"
                f"{kl_label}: {row[kl_col]:.4f}<br>"
                f"N Questions: {row['count']}"
            )
            hover_text.append(text)

        fig.add_trace(
            go.Scatter(
                x=df_cat[metric],
                y=df_cat[kl_col],
                mode="markers+text",
                name=f"{category} (n={len(df_cat)})",
                marker=dict(
                    size=df_cat["count"].apply(lambda x: min(40, 8 + x * 0.02)),
                    color=color_map[category],
                    line=dict(width=1, color="black"),
                    opacity=0.7,
                ),
                text=df_cat["subject"].apply(
                    lambda x: x.replace("_", " ").title()
                ),
                textposition="top center",
                textfont=dict(size=8),
                hovertext=hover_text,
                hoverinfo="text",
            )
        )

    # Calculate Spearman correlation and add trend line
    spearman_r, spearman_p = scipy_stats.spearmanr(
        subject_stats[metric], subject_stats[kl_col], nan_policy="omit"
    )

    # Drop NaN values for linear regression to match spearmanr behavior
    valid_mask = ~(subject_stats[metric].isna() | subject_stats[kl_col].isna())
    valid_x = subject_stats[metric][valid_mask]
    valid_y = subject_stats[kl_col][valid_mask]

    # Still plot linear trend line for visualization
    if len(valid_x) > 1:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            valid_x, valid_y
        )
    else:
        # If not enough valid data, use zeros
        slope, intercept, r_value, p_value, std_err = 0, 0, 0, 1, 0
    x_trend = np.linspace(
        subject_stats[metric].min(), subject_stats[metric].max(), 100
    )
    y_trend = slope * x_trend + intercept

    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode="lines",
            name=f"Trend (Spearman r={spearman_r:.3f})",
            line=dict(color="black", width=2, dash="dash"),
            hovertemplate=f"Trend line<br>Spearman r={spearman_r:.3f}<br>p-value={spearman_p:.3f}<extra></extra>",
        )
    )

    # Format x-axis based on metric
    x_format = ".0%" if metric != "thinking_improvement" else ".1%"

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Subject Performance: {metric_label} vs {kl_label}",
            font=dict(size=18, family="Arial, sans-serif"),
        ),
        xaxis=dict(
            title=dict(text=metric_label, font=dict(size=14)),
            tickformat=x_format,
            gridcolor="lightgray",
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text=kl_label, font=dict(size=14)),
            gridcolor="lightgray",
            showgrid=True,
        ),
        hovermode="closest",
        template="plotly_white",
        width=1200,
        height=800,
        legend=dict(
            title="Range",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )

    # Save to HTML
    metric_short = (
        metric.replace("_", "")
        .replace("accuracy", "acc")
        .replace("improvement", "imp")
    )
    model_dir = f"results/{model_name.replace('/', '-')}"
    os.makedirs(model_dir, exist_ok=True)
    output_path = f"{model_dir}/{metric_short}_vs_{kl_type}_kl_interactive_rc{require_correct}.html"
    fig.write_html(output_path)
    print(
        f"Saved interactive {metric_label} vs {kl_type} KL scatter plot to {output_path}"
    )
