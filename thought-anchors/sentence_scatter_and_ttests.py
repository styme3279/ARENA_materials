import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def paired_ttests(df, key="counterfactual_importance_kl"):
    df = df[df["is_converged"] == False]
    # df["abs_delta_acc"] = df["delta_acc"].abs()
    # df["abs_delta_logit"] = df["delta_logit"].abs()
    # key = "abs_delta_acc"

    df[key] = df[key].abs()

    df_grp = (
        df.groupby(["problem_idx", "function_tags"])[key].median().reset_index()
    )

    for tag0, df_tag0 in df_grp.groupby("function_tags"):
        for tag1, df_tag1 in df_grp.groupby("function_tags"):
            if tag0 == tag1:
                continue
            if "final_answer" in tag0 or "final_answer" in tag1:
                continue
            if "setup" in tag0 or "setup" in tag1:
                continue
            if "self_checking" in tag0 or "self_checking" in tag1:
                continue
            overlapping_problems = set(df_tag0["problem_idx"]) & set(
                df_tag1["problem_idx"]
            )
            df_tag0_ = df_tag0[
                df_tag0["problem_idx"].isin(overlapping_problems)
            ]
            df_tag1_ = df_tag1[
                df_tag1["problem_idx"].isin(overlapping_problems)
            ]
            # print(df_tag0_)

            df_tag0_.set_index("problem_idx", inplace=True)
            df_tag1_.set_index("problem_idx", inplace=True)

            t, p = stats.ttest_rel(
                df_tag0_[key], df_tag1_[key], nan_policy="omit"
            )
            M_0 = df_tag0_[key].mean()
            M_1 = df_tag1_[key].mean()
            print(
                f"{tag0} ({M_0:.3f}) vs {tag1} ({M_1:.3f}): t={t:.2f}, p={p:.3f}"
            )


def plot_scatter(df, key="counterfactual_importance_kl", output_dir="plots/pb"):
    """
    Create scatter plot showing importance vs position for different categories.
    Uses mean and SE calculated from the data.

    Args:
        df: DataFrame with chunk data
        key: Importance metric to plot
        output_dir: Directory to save plots
    """
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Filter and prepare data
    df = df[df["is_converged"] == False]
    # df.loc[df["is_converged"] == False, key] = np.nan
    df[key] = df[key].abs()
    df["function_tags"] = df["function_tags"].apply(
        lambda x: " ".join(word.capitalize() for word in x.split("_"))
    )
    categories = {
        "Active Computation",
        "Fact Retrieval",
        "Plan Generation",
        "Uncertainty Management",
        "Result Consolidation",
    }
    df = df[df["function_tags"].isin(categories)]

    # Calculate per-problem medians first, then aggregate
    df_grp = (
        df.groupby(["problem_idx", "function_tags"])[
            [key, "normalized_position"]
        ]
        .median()
        .reset_index()
    )

    # Calculate means and SEs across problems
    df_Ms = (
        df_grp.groupby("function_tags")[[key, "normalized_position"]]
        .mean()
        .reset_index()
    )
    df_SEs = (
        df_grp.groupby("function_tags")[[key, "normalized_position"]]
        .sem()  # Use SEM instead of std for error bars
        .reset_index()
    )
    # print(f"{df_Ms=}, {df_SEs=}")
    # quit()

    # Define category colors
    CATEGORY_COLORS = {
        "Active Computation": "#34A853",
        "Fact Retrieval": "#FBBC05",
        "Final Answer Emission": "#795548",
        "Plan Generation": "#EA4335",
        "Problem Setup": "#4285F4",
        "Result Consolidation": "#00BCD4",
        "Self Checking": "#FF9800",
        "Uncertainty Management": "#9C27B0",
    }

    # Determine plot title based on key
    if "resampling" in key:
        plot_type = "Resampling"
    elif "counterfactual" in key:
        plot_type = "Counterfactual"
    elif "forced" in key:
        plot_type = "Forced"
    else:
        plot_type = ""

    if "accuracy" in key:
        measure = "Accuracy"
    elif "kl" in key:
        measure = "KL"
    else:
        measure = ""

    FONT_SIZE = 20
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE + 4,
            "axes.labelsize": FONT_SIZE + 2,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE - 1,
            "figure.titlesize": FONT_SIZE + 20,
        }
    )

    # Create figure
    plt.figure(figsize=(9, 7))

    # Plot each category
    for category in categories:
        # Get data for this category
        cat_mean = df_Ms[df_Ms["function_tags"] == category]
        cat_se = df_SEs[df_SEs["function_tags"] == category]

        if cat_mean.empty:
            continue

        # Get color
        cat_color = CATEGORY_COLORS.get(category, "#7f7f7f")

        # Plot with error bars
        plt.errorbar(
            cat_mean["normalized_position"].values[0],
            cat_mean[key].values[0],
            xerr=cat_se["normalized_position"].values[0],
            yerr=cat_se[key].values[0],
            fmt="o",
            markersize=10,
            alpha=0.7,
            capsize=5,
            label=category,
            color=cat_color,
            linewidth=1.5,
        )

    # Remove top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Set labels and title
    plt.xlabel("Normalized position in trace (0-1)", labelpad=20)
    plt.ylabel(f"{plot_type} importance ({measure})", labelpad=20)
    plt.title(f"Sentence category effect")

    # Set x-axis range
    # plt.xlim(-0.05, 1.05)

    # Calculate y-axis range to zoom in
    y_values = df_Ms[key]
    y_errors = df_SEs[key]
    # print(f"{y_values=}, {y_errors=}")
    # quit()
    y_min = (y_values - y_errors).min()
    y_max = (y_values + y_errors).max()
    y_range = y_max - y_min
    plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    x_values = df_Ms["normalized_position"]
    x_errors = df_SEs["normalized_position"]
    x_min = (x_values - x_errors).min()
    x_max = (x_values + x_errors).max()
    x_range = x_max - x_min
    plt.xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)

    # Add legend
    # plt.legend(frameon=False, fontsize=12)

    # Tight layout
    plt.tight_layout()
    # plt.show()
    # quit()

    # Save plot
    plot_filename = f"{model}_alpha_{smoothing}_{key}_category_effects.png"
    plt.savefig(
        os.path.join(output_dir, plot_filename), dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Created plot for {key}: {plot_filename}")


if __name__ == "__main__":
    model = "deepseek-r1-distill-qwen-14b"
    smoothing = 0.5
    fp_json = rf"analysis\basic\{model}\alpha_{smoothing}\incorrect_base_solution\analysis_results.json"
    with open(fp_json, "r") as f:
        data = json.load(f)
    fp_json_correct = rf"analysis\basic\{model}\alpha_{smoothing}\correct_base_solution\analysis_results.json"
    with open(fp_json_correct, "r") as f:
        data_correct = json.load(f)
    data = data + data_correct

    df_as_l = []
    convergence_check = 5
    convergence_threshold = 0.95
    convergence_cnt = 0

    problem_idx_to_convergence = {}

    for problem_data in tqdm(data, desc="Making df"):
        prev_acc = None
        delta_acc = None
        base_correct = problem_data["base_accuracy"]
        total_chunks = len(problem_data["labeled_chunks"])
        is_converged = False
        for chunk_idx, chunk in enumerate(problem_data["labeled_chunks"]):
            acc = chunk["accuracy"]
            if acc > convergence_threshold or acc < 1 - convergence_threshold:
                if not is_converged:
                    convergence_cnt += 1
                    if convergence_cnt >= convergence_check:
                        is_converged = True
                        problem_idx_to_convergence[
                            problem_data["problem_idx"]
                        ] = convergence_cnt
            else:
                convergence_cnt = 0

            acc_for_logit = min(max(acc, 0.01), 0.99)
            logit = np.log(acc_for_logit / (1 - acc_for_logit))
            if prev_acc is not None:
                prev_acc_for_logit = min(max(prev_acc, 0.01), 0.99)
                prev_logit = np.log(
                    prev_acc_for_logit / (1 - prev_acc_for_logit)
                )
                delta_acc = acc - prev_acc
                delta_logit = logit - prev_logit
            else:
                delta_acc = None
                delta_logit = None
            prev_acc = acc
            row = {}
            row["problem_idx"] = (
                problem_data["problem_idx"] + f"_{base_correct}"
            )
            row["delta_acc"] = delta_acc
            row["delta_logit"] = delta_logit
            row["function_tags"] = chunk["function_tags"][0]
            row["acc"] = acc
            row["is_converged"] = is_converged
            row["normalized_position"] = (
                chunk_idx / (total_chunks - 1) if total_chunks > 1 else 0.5
            )
            row["chunk_idx"] = chunk_idx
            keys = [
                "resampling_importance_kl",
                "resampling_importance_accuracy",
                "forced_importance_kl",
                "forced_importance_accuracy",
                "counterfactual_importance_kl",
                "counterfactual_importance_accuracy",
            ]
            for key in keys:
                row[key] = chunk[key]
            df_as_l.append(row)

    df = pd.DataFrame(df_as_l)
    df["idx"] = df.index
    # df['convergence_cnt'] = df['problem_idx'].map(problem_idx_to_convergence)
    # paired_ttests(df)
    # paired_ttests(df, key="forced_importance_kl")

    plot_scatter(df, key="counterfactual_importance_kl")
    plot_scatter(df, key="forced_importance_kl")