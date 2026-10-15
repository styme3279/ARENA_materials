import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects

# Global font size for plots
FONT_SIZE = 16
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 2,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 2,
    'ytick.labelsize': FONT_SIZE - 2,
    'legend.fontsize': FONT_SIZE - 2,
})

def load_data(base_dir: Path, solution_type: str) -> List[Dict]:
    """
    Load step importance and chunk text data for all problems of a specific solution type.
    
    Args:
        base_dir: Base directory containing step attribution data
        solution_type: Type of solution (correct_base_solution, incorrect_base_solution)
    
    Returns:
        List of problem data with importance scores and categories
    """
    solution_dir = base_dir / solution_type
    all_data = []
    
    if not solution_dir.exists():
        print(f"Directory not found: {solution_dir}")
        return all_data
    
    problem_dirs = [d for d in solution_dir.iterdir() if d.is_dir()]
    
    for problem_dir in tqdm(problem_dirs, desc=f"Loading {solution_type} data"):
        importance_file = problem_dir / "step_importance.json"
        chunk_texts_file = problem_dir / "chunk_texts.json"
        
        if importance_file.exists() and chunk_texts_file.exists():
            try:
                # Load importance data
                with open(importance_file, 'r', encoding='utf-8') as f:
                    importance_data = json.load(f)
                
                # Load chunk texts data
                with open(chunk_texts_file, 'r', encoding='utf-8') as f:
                    chunk_texts = json.load(f)
                
                # Create a mapping from chunk_idx to category (first function tag)
                chunk_categories = {}
                for chunk in chunk_texts:
                    chunk_idx = chunk["chunk_idx"]
                    category = chunk["function_tags"][0] if chunk["function_tags"] else "unknown"
                    chunk_categories[chunk_idx] = category
                
                # Add categories to importance data
                problem_data = {
                    "problem_id": problem_dir.name,
                    "importance_data": importance_data,
                    "chunk_categories": chunk_categories
                }
                
                all_data.append(problem_data)
            except Exception as e:
                print(f"Error loading data from {problem_dir}: {e}")
    
    return all_data

def identify_steps_with_importance(problem_data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Identify all steps with their importance scores (no threshold filtering).
    
    Args:
        problem_data: List of problem data with importance scores
    
    Returns:
        Dictionary mapping problem_id to list of steps with importance
    """
    steps_with_importance = {}
    
    for problem in problem_data:
        problem_id = problem["problem_id"]
        importance_data = problem["importance_data"]
        chunk_categories = problem["chunk_categories"]
        
        # Collect all steps with their importance
        all_steps = []
        for step in importance_data:
            source_idx = step["source_chunk_idx"]
            source_category = chunk_categories.get(source_idx, "unknown")
            
            for target in step["target_impacts"]:
                target_idx = target["target_chunk_idx"]
                target_category = chunk_categories.get(target_idx, "unknown")
                score = target["importance_score"]
                
                if score > 0 or True:
                    all_steps.append({
                        "source_idx": source_idx,
                        "source_category": source_category,
                        "target_idx": target_idx,
                        "target_category": target_category,
                        "importance_score": score
                    })
        
        steps_with_importance[problem_id] = all_steps
    
    return steps_with_importance

from collections import defaultdict
CATEGORIES = defaultdict(list)

def analyze_transitions(steps_with_importance: Dict[str, List[Dict]]) -> Tuple[Dict, Dict]:
    """
    Analyze transitions for steps.
    
    Args:
        steps_with_importance: Dictionary mapping problem_id to list of steps with importance
    
    Returns:
        Tuple of (incoming_categories, outgoing_categories) dictionaries
    """
    outgoing_categories = defaultdict(list)
    
    for problem_id, steps in steps_with_importance.items():
        for step in steps:
            source_category = step["source_category"]
            target_category = step["target_category"]
            score = step["importance_score"]
            
            # Record outgoing transition (source -> target)
            outgoing_categories[source_category].append({
                "target_category": target_category,
                "importance_score": score,
                "problem_id": problem_id
            })
    
    # Return empty dict for incoming to maintain compatibility
    return {}, outgoing_categories

def calculate_transition_stats(transitions: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Calculate statistics for transitions.
    
    Args:
        transitions: Dictionary mapping categories to lists of transitions
        
    Returns:
        DataFrame with transition statistics
    """
    stats = []
    
    for source_category, category_transitions in transitions.items():
        # Count transitions by target category
        transition_counts = Counter()
        transition_scores = defaultdict(list)
        
        for transition in category_transitions:
            target_category = transition["target_category"]
            transition_counts[target_category] += 1
            transition_scores[target_category].append(transition["importance_score"])
        
        # Calculate statistics for each transition category
        total_transitions = sum(transition_counts.values())
        
        for target_category, count in transition_counts.items():
            # Calculate percentage (normalize within each category)
            percentage = (count / total_transitions) * 100
            
            # Calculate average importance score for this transition
            scores = transition_scores[target_category]
            avg_score = np.mean(scores) if scores else 0
            
            # Add to statistics
            stats.append({
                "source_category": source_category,
                "target_category": target_category,
                "count": count,
                "percentage": percentage,
                "avg_importance": avg_score
            })
    
    # Convert to DataFrame
    if stats:
        return pd.DataFrame(stats)
    else:
        return pd.DataFrame(columns=["source_category", "target_category", "count", "percentage", "avg_importance"])

def format_category_name(category: str) -> str:
    """Format category name for better display."""
    return " ".join(word.capitalize() for word in category.split("_"))

def plot_transition_heatmaps(incoming_stats: pd.DataFrame, outgoing_stats: pd.DataFrame, 
                            solution_type: str, output_dir: Path, top_n: int = 8):
    """
    Plot heatmaps for incoming and outgoing transitions.
    
    Args:
        incoming_stats: DataFrame with incoming transition statistics
        outgoing_stats: DataFrame with outgoing transition statistics
        solution_type: Type of solution (all, correct, incorrect)
        output_dir: Directory to save plots
        top_n: Number of top categories to include in each heatmap
    """
    # Check if dataframes are empty
    if isinstance(incoming_stats, pd.DataFrame):
        incoming_empty = incoming_stats.empty
    else:
        incoming_empty = not bool(incoming_stats)  # Empty dict will be False
        
    if isinstance(outgoing_stats, pd.DataFrame):
        outgoing_empty = outgoing_stats.empty
    else:
        outgoing_empty = not bool(outgoing_stats)  # Empty dict will be False
    
    if incoming_empty and outgoing_empty:
        print(f"No data available for {solution_type} solutions")
        return
    
    # Create figure with one subplot (only outgoing)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot outgoing transitions
    if not outgoing_empty:
        plot_heatmap(outgoing_stats, "Outgoing", solution_type, ax, top_n)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f"{solution_type}_transitions_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap for {solution_type} solutions to {output_path}")

def plot_heatmap(stats: pd.DataFrame, direction: str, solution_type: str, ax, top_n: int = 8):
    """
    Plot a single heatmap for transitions.
    
    Args:
        stats: DataFrame with transition statistics
        direction: Direction of transitions ("Incoming" or "Outgoing")
        solution_type: Type of solution (all, correct, incorrect)
        ax: Matplotlib axis to plot on
        top_n: Number of top categories to include
    """
    # Get top categories by total percentage
    if direction == "Incoming":
        source_col = "source_category"
        target_col = "target_category"
    else:  # Outgoing
        source_col = "source_category"
        target_col = "target_category"
    
    # Get top source categories
    top_sources = (stats.groupby(source_col)['percentage'].sum()
                  .sort_values(ascending=False).head(top_n).index.tolist())
    
    # Get top target categories
    top_targets = (stats.groupby(target_col)['percentage'].sum()
                  .sort_values(ascending=False).head(top_n).index.tolist())
    
    # Filter for top categories
    filtered_stats = stats[
        (stats[source_col].isin(top_sources)) & 
        (stats[target_col].isin(top_targets))
    ]
    
    # Format category names
    filtered_stats[source_col] = filtered_stats[source_col].apply(format_category_name)
    filtered_stats[target_col] = filtered_stats[target_col].apply(format_category_name)
    
    # Create pivot table
    pivot_data = filtered_stats.pivot_table(
        index=source_col, 
        columns=target_col, 
        values='percentage',
        fill_value=0
    )
    
    # Plot heatmap
    sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, 
               cbar_kws={'label': 'Percentage (%)'})
    
    # Set title and labels
    ax.set_title(f"Top {direction} Transitions - {solution_type.capitalize()} Solutions", fontsize=FONT_SIZE+4)
    ax.set_ylabel(f"Source Category", fontsize=FONT_SIZE)
    ax.set_xlabel(f"Target Category", fontsize=FONT_SIZE)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")

def plot_summary_barplots(all_stats: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
                         output_dir: Path, top_n: int = 10):
    """
    Create summary bar plots comparing transitions across solution types.
    
    Args:
        all_stats: Dictionary mapping solution type to (incoming_stats, outgoing_stats)
        output_dir: Directory to save plots
        top_n: Number of top categories to include
    """
    # Prepare data for summary plots
    summary_data = []
    
    for solution_type, (incoming_stats, outgoing_stats) in all_stats.items():
        # Check if dataframes are empty
        if isinstance(incoming_stats, pd.DataFrame):
            incoming_empty = incoming_stats.empty
        else:
            incoming_empty = not bool(incoming_stats)  # Empty dict will be False
            
        if isinstance(outgoing_stats, pd.DataFrame):
            outgoing_empty = outgoing_stats.empty
        else:
            outgoing_empty = not bool(outgoing_stats)  # Empty dict will be False
            
        if incoming_empty or outgoing_empty:
            continue
            
        # Aggregate outgoing transitions
        if 'source_category' in outgoing_stats.columns:
            # Group by source_category and calculate mean percentage
            outgoing_agg = (outgoing_stats.groupby('source_category')['percentage'].mean()
                          .reset_index().rename(columns={'percentage': 'total_percentage'}))
            outgoing_agg['direction'] = 'Outgoing'
            outgoing_agg['solution_type'] = solution_type.capitalize()
            outgoing_agg['category'] = outgoing_agg['source_category'].apply(format_category_name)
            summary_data.append(outgoing_agg)
    
    if not summary_data:
        print("No data available for summary plots")
        return
        
    # Combine all data
    df_summary = pd.concat(summary_data, ignore_index=True)
    
    # Get top categories across all solution types
    top_categories = (df_summary.groupby('category')['total_percentage']
                    .sum().sort_values(ascending=False).head(top_n).index.tolist())
    
    # Filter for top categories
    df_summary_filtered = df_summary[df_summary['category'].isin(top_categories)]
    
    # Create summary plot
    g = sns.catplot(
        data=df_summary_filtered,
        kind="bar",
        x="category", y="total_percentage",
        hue="solution_type", col="direction",
        height=8, aspect=1.2, palette="viridis",
        legend_out=False
    )
    
    # Customize plot
    g.set_xticklabels(rotation=60, ha="right")
    g.set_axis_labels("Category", "Mean Percentage (%)")
    g.fig.suptitle("Summary of Step Transitions by Solution Type", fontsize=FONT_SIZE+6)
    g.fig.subplots_adjust(top=0.9)
    
    # Save figure
    output_path = output_dir / "step_transitions_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plot to {output_path}")
    
    # Also save as CSV
    df_summary.to_csv(output_dir / "step_transitions_summary.csv", index=False)

def plot_average_importance(all_stats: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
                           output_dir: Path, use_abs: bool = False, top_n: int = 10):
    """
    Plot average importance scores for each source category.
    
    Args:
        all_stats: Dictionary mapping solution type to (incoming_stats, outgoing_stats)
        output_dir: Directory to save plots
        use_abs: Whether to use absolute values of importance scores
        top_n: Number of top categories to include
    """
    # Prepare data for importance plots
    importance_data = []
    
    for solution_type, (_, outgoing_stats) in all_stats.items():
        # Skip "all" solution type for these plots
        if solution_type == "all":
            continue
            
        # Check if dataframe is empty
        if isinstance(outgoing_stats, pd.DataFrame):
            if outgoing_stats.empty:
                continue
        elif not outgoing_stats:  # Empty dict
            continue
            
        # Process importance scores
        if 'source_category' in outgoing_stats.columns and 'avg_importance' in outgoing_stats.columns:
            # Keep all individual data points for error bar calculation
            importance_df = outgoing_stats.copy()
            if use_abs:
                importance_df['avg_importance'] = importance_df['avg_importance'].abs()
                
            importance_df['solution_type'] = solution_type.capitalize()
            importance_df['category'] = importance_df['source_category'].apply(format_category_name)
            
            # Add count column for proper sample size calculation
            importance_data.append(importance_df)
    
    if not importance_data:
        print("No data available for importance plots")
        return
        
    # Combine all data
    df_importance = pd.concat(importance_data, ignore_index=True)
    
    # Create importance plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for solution types
    colors = {'Correct': 'green', 'Incorrect': 'red'}
    
    # Calculate average importance per category (across solution types)
    avg_by_category = df_importance.groupby('category')['avg_importance'].mean().reset_index()
    
    # Sort categories by average importance
    sorted_categories = avg_by_category.sort_values('avg_importance', ascending=False)['category'].tolist()
    
    # Limit to top N categories
    top_categories = sorted_categories[:top_n]
    
    # Filter data for top categories
    plot_data = df_importance[df_importance['category'].isin(top_categories)]
    
    # Create a categorical type with the right order
    plot_data['category'] = pd.Categorical(
        plot_data['category'], 
        categories=top_categories,
        ordered=True
    )
    
    # Create bar plot with seaborn's default error bars
    sns.barplot(
        data=plot_data,
        x='category',
        y='avg_importance',
        hue='solution_type',
        palette=colors,
        errorbar=('ci', 95),  # Explicitly set 95% confidence interval
        ax=ax
    )
    
    # Calculate sample sizes for each category and solution type
    # Sum the 'count' column to get the actual number of steps
    sample_sizes = df_importance.groupby(['category', 'solution_type'])['count'].sum().reset_index()
    
    # Print sample sizes for debugging
    print("Sample sizes:")
    print(sample_sizes)
    
    # Add sample sizes as text annotations inside the bars
    for i, patch in enumerate(ax.patches):
        # Get the category and solution type for this patch
        category_idx = i // 2  # Integer division to get category index
        solution_idx = i % 2   # Remainder to get solution type (0=Correct, 1=Incorrect)
        
        if category_idx < len(top_categories):
            category = top_categories[category_idx]
            solution_type = ['Correct', 'Incorrect'][solution_idx]
            
            # Find the corresponding sample size
            size_row = sample_sizes[(sample_sizes['category'] == category) & 
                                   (sample_sizes['solution_type'] == solution_type)]
            
            if not size_row.empty:
                n = int(size_row['count'].values[0])
                
                # Position the text in the middle of the bar
                bar_height = patch.get_height()
                x_pos = patch.get_x() + patch.get_width() / 2
                y_pos = bar_height / 2  # Middle of the bar
                
                # Add the annotation with white text and black outline for visibility
                text = ax.text(x_pos, y_pos, f"n={n}", 
                       ha='center', va='center', fontsize=10, 
                       color='white', fontweight='bold')
                text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
    
    # Customize plot
    ax.set_title('Average Importance by Source Category', fontsize=FONT_SIZE+2)
    ax.set_ylabel('Average Importance Score', fontsize=FONT_SIZE)
    ax.set_xlabel('Source Category', fontsize=FONT_SIZE)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title=None)
    
    # Add overall title
    abs_text = "Absolute " if use_abs else ""
    plt.suptitle(f'Average {abs_text}Importance Scores by Source Category', fontsize=FONT_SIZE+4)
    plt.tight_layout()
    
    # Save figure
    abs_suffix = "_abs" if use_abs else ""
    output_path = output_dir / f"average_importance{abs_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved average importance plot to {output_path}")
    
    # Also save as CSV
    df_importance.to_csv(output_dir / f"average_importance{abs_suffix}.csv", index=False)

def plot_total_impact(all_stats: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
                     output_dir: Path, use_abs: bool = False, top_n: int = 10):
    """
    Plot total impact scores (avg_importance × count) for each source category.
    
    Args:
        all_stats: Dictionary mapping solution type to (incoming_stats, outgoing_stats)
        output_dir: Directory to save plots
        use_abs: Whether to use absolute values of importance scores
        top_n: Number of top categories to include
    """
    # Prepare data for impact plots
    impact_data = []
    
    for solution_type, (_, outgoing_stats) in all_stats.items():
        # Skip "all" solution type for these plots
        if solution_type == "all":
            continue
            
        # Check if dataframe is empty
        if isinstance(outgoing_stats, pd.DataFrame):
            if outgoing_stats.empty:
                continue
        elif not outgoing_stats:  # Empty dict
            continue
            
        # Process impact scores
        if 'source_category' in outgoing_stats.columns and 'avg_importance' in outgoing_stats.columns:
            # Keep all individual data points for error bar calculation
            impact_df = outgoing_stats.copy()
            if use_abs:
                impact_df['avg_importance'] = impact_df['avg_importance'].abs()
            
            # Calculate impact for each row
            impact_df['impact'] = impact_df['avg_importance'] * impact_df['count']
            impact_df['solution_type'] = solution_type.capitalize()
            impact_df['category'] = impact_df['source_category'].apply(format_category_name)
            impact_data.append(impact_df)
    
    if not impact_data:
        print("No data available for impact plots")
        return
        
    # Combine all data
    df_impact = pd.concat(impact_data, ignore_index=True)
    
    # Create impact plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for solution types
    colors = {'Correct': 'green', 'Incorrect': 'red'}
    
    # Calculate average impact per category (across solution types)
    avg_by_category = df_impact.groupby('category')['impact'].mean().reset_index()
    
    # Sort categories by average impact
    sorted_categories = avg_by_category.sort_values('impact', ascending=False)['category'].tolist()
    
    # Limit to top N categories
    top_categories = sorted_categories[:top_n]
    
    # Filter data for top categories
    plot_data = df_impact[df_impact['category'].isin(top_categories)]
    
    # Create a categorical type with the right order
    plot_data['category'] = pd.Categorical(
        plot_data['category'], 
        categories=top_categories,
        ordered=True
    )
    
    # Create bar plot with error bars
    sns.barplot(
        data=plot_data,
        x='category',
        y='impact',
        hue='solution_type',
        palette=colors,
        errorbar=('ci', 95),  # Add 95% confidence interval
        ax=ax
    )
    
    # Customize plot
    ax.set_title('Total Impact by Source Category', fontsize=FONT_SIZE+2)
    ax.set_ylabel('Impact (Importance × Count)', fontsize=FONT_SIZE)
    ax.set_xlabel('Source Category', fontsize=FONT_SIZE)
    ax.tick_params(axis='x', rotation=60)
    ax.grid(axis='y', alpha=0.3)
    
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title=None)
    
    # Add overall title
    abs_text = "Absolute " if use_abs else ""
    plt.tight_layout()
    
    # Save figure
    abs_suffix = "_abs" if use_abs else ""
    output_path = output_dir / f"total_impact{abs_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved total impact plot to {output_path}")
    
    # Also save as CSV
    df_impact.to_csv(output_dir / f"total_impact{abs_suffix}.csv", index=False)

def main():
    parser = argparse.ArgumentParser(description="Analyze step transitions in problem solutions")
    parser.add_argument("--base_dir", type=str, default="analysis/step_attribution",
                        help="Base directory containing step attribution data")
    parser.add_argument("--output_dir", type=str, default="analysis/step_transitions",
                        help="Directory to save analysis results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define solution types to analyze
    solution_types = {
        "correct": ["correct_base_solution"],
        "incorrect": ["incorrect_base_solution"],
        "all": ["correct_base_solution", "incorrect_base_solution"]
    }
    
    # Store data for each solution type
    all_problem_data = {}
    
    # Load data for each solution type
    for solution_type, directories in solution_types.items():
        print(f"\nProcessing {solution_type} solutions...")
        
        # Combine data from all directories for this solution type
        combined_data = []
        
        for directory in directories:
            print(f"Loading data from {directory}...")
            data = load_data(Path(args.base_dir), directory)
            combined_data.extend(data)
        
        all_problem_data[solution_type] = combined_data
    
    # Process each solution type
    all_stats = {}
    
    for solution_type, problem_data in all_problem_data.items():
        print(f"\nAnalyzing {solution_type} solutions...")
        
        # Identify all steps with importance (no threshold)
        steps_with_importance = identify_steps_with_importance(problem_data)
        
        # Analyze transitions (only care about outgoing now)
        _, outgoing_categories = analyze_transitions(steps_with_importance)
        
        # Calculate statistics
        outgoing_stats = calculate_transition_stats(outgoing_categories)
        
        # Store statistics (empty dict for incoming to maintain compatibility)
        all_stats[solution_type] = ({}, outgoing_stats)
        
        # Plot heatmaps (only for outgoing)
        plot_transition_heatmaps({}, outgoing_stats, solution_type, output_dir)
    
    # Create summary plots (only for outgoing)
    plot_summary_barplots(all_stats, output_dir)
    
    # Create average importance plots (only for correct and incorrect)
    plot_average_importance(all_stats, output_dir, use_abs=False)
    plot_average_importance(all_stats, output_dir, use_abs=True)
    
    # Create total impact plots (new)
    plot_total_impact(all_stats, output_dir, use_abs=False)
    plot_total_impact(all_stats, output_dir, use_abs=True)
    
    for category, scores in CATEGORIES.items():
        print(category, len(scores), np.mean(scores), np.std(scores))
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()