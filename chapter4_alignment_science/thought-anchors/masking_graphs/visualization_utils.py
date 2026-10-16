"""
Interactive visualization utilities for MMLU analysis.

This module provides functions to create interactive HTML visualizations
using plotly for better exploration of the analysis results.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def create_interactive_subject_scatter(
    thinking_df: pd.DataFrame,
    output_path: str = "results/analysis/subject_scatter_interactive.html",
    min_questions: int = 10
) -> None:
    """
    Create an interactive scatter plot comparing thinking vs non-thinking accuracy by subject.
    
    Args:
        thinking_df: DataFrame with paired thinking/non-thinking results
        output_path: Where to save the HTML file
        min_questions: Minimum number of questions to include a subject
    """
    if "subject" not in thinking_df.columns:
        print("No subject column found in data")
        return
    
    # Calculate accuracy for each subject
    subject_data = []
    for subject in thinking_df["subject"].unique():
        subset = thinking_df[thinking_df["subject"] == subject]
        if len(subset) >= min_questions:
            nonthinking_acc = subset["is_correct"].mean()
            thinking_acc = subset["thinking_is_correct"].mean()
            improvement = thinking_acc - nonthinking_acc
            
            # Determine category for coloring
            if improvement > 0.1:
                category = "Large Improvement (>10%)"
                color = "#2ecc71"
            elif improvement > 0:
                category = "Small Improvement (0-10%)"
                color = "#3498db"
            elif improvement > -0.05:
                category = "Small Degradation (0-5%)"
                color = "#f39c12"
            else:
                category = "Large Degradation (>5%)"
                color = "#e74c3c"
            
            subject_data.append({
                "subject": subject,
                "subject_display": subject.replace("_", " ").title(),
                "nonthinking": nonthinking_acc,
                "thinking": thinking_acc,
                "count": len(subset),
                "improvement": improvement,
                "improvement_pct": improvement * 100,
                "category": category,
                "color": color
            })
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame(subject_data)
    
    # Sort by improvement for consistent ordering
    plot_df = plot_df.sort_values("improvement", ascending=False)
    
    # Find min/max for axis ranges
    all_accuracies = list(plot_df["nonthinking"]) + list(plot_df["thinking"])
    min_acc = min(all_accuracies) - 0.05
    max_acc = max(all_accuracies) + 0.05
    
    # Create the scatter plot
    fig = go.Figure()
    
    # Add traces for each category
    for category in plot_df["category"].unique():
        df_cat = plot_df[plot_df["category"] == category]
        
        # Determine marker symbol
        if "Improvement" in category:
            symbol = "triangle-up"
        else:
            symbol = "triangle-down"
        
        fig.add_trace(go.Scatter(
            x=df_cat["nonthinking"],
            y=df_cat["thinking"],
            mode='markers+text',
            name=category,
            marker=dict(
                size=df_cat["count"].apply(lambda x: min(30, 5 + x * 0.02)),
                color=df_cat["color"],
                symbol=symbol,
                line=dict(width=1, color='black')
            ),
            text=df_cat["subject_display"],
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Non-thinking: %{x:.1%}<br>" +
                "Thinking: %{y:.1%}<br>" +
                "Improvement: %{customdata[0]:.1f}%<br>" +
                "Questions: %{customdata[1]}<br>" +
                "<extra></extra>"
            ),
            customdata=np.column_stack((df_cat["improvement_pct"], df_cat["count"])),
            showlegend=True
        ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[min_acc, max_acc],
        y=[min_acc, max_acc],
        mode='lines',
        name='Equal Performance',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Subject Performance: Thinking vs Non-thinking Mode<br>" +
                   f"<sub>Based on {len(thinking_df)} paired questions across {len(plot_df)} subjects</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis=dict(
            title="Non-thinking Accuracy",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            tickformat=".0%",
            range=[min_acc, max_acc],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            title="Thinking Accuracy",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            tickformat=".0%",
            range=[min_acc, max_acc],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            scaleanchor="x",
            scaleratio=1
        ),
        height=800,
        width=800,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Black",
            borderwidth=1
        )
    )
    
    # Add annotations for statistics
    improvements = plot_df["improvement"].values
    avg_improvement = np.mean(improvements) * 100
    num_improved = sum(1 for i in improvements if i > 0)
    num_degraded = sum(1 for i in improvements if i < 0)
    
    stats_text = (f"Avg improvement: {avg_improvement:.1f}%<br>"
                 f"Subjects improved: {num_improved}/{len(plot_df)}<br>"
                 f"Subjects degraded: {num_degraded}/{len(plot_df)}")
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12),
        align="left"
    )
    
    # Save the figure
    fig.write_html(output_path)
    print(f"Interactive scatter plot saved to {output_path}")


def create_interactive_difficulty_plot(
    paired_df: pd.DataFrame,
    confidence_col: str = "prob_generated",
    output_path: str = "results/analysis/difficulty_interactive.html"
) -> None:
    """
    Create an interactive bar plot showing performance by difficulty level.
    
    Args:
        paired_df: DataFrame with paired results
        confidence_col: Column to use for difficulty binning
        output_path: Where to save the HTML file
    """
    # Create difficulty bins
    difficulty_bins = [0, 0.25, 0.5, 0.75, 1.0]
    paired_df = paired_df.copy()
    
    # Check if confidence column exists
    if confidence_col not in paired_df.columns:
        print(f"Warning: {confidence_col} not found, using prob_correct as fallback")
        confidence_col = "prob_correct"
    
    paired_df["difficulty_bin"] = pd.cut(
        paired_df[confidence_col],
        bins=difficulty_bins,
        labels=["Hard (<25%)", "Medium-Hard (25-50%)", "Medium-Easy (50-75%)", "Easy (75-100%)"],
        include_lowest=True
    )
    
    # Calculate statistics for each difficulty level
    difficulty_data = []
    for difficulty in paired_df["difficulty_bin"].cat.categories:
        subset = paired_df[paired_df["difficulty_bin"] == difficulty]
        if len(subset) > 0:
            difficulty_data.append({
                "Difficulty": difficulty,
                "Non-thinking": subset["is_correct"].mean(),
                "Thinking": subset["thinking_is_correct"].mean(),
                "Count": len(subset),
                "Improvement": subset["thinking_is_correct"].mean() - subset["is_correct"].mean()
            })
    
    if not difficulty_data:
        print("No data for difficulty plot")
        return
    
    diff_df = pd.DataFrame(difficulty_data)
    
    # Create the figure
    fig = go.Figure()
    
    # Add bars for non-thinking accuracy
    fig.add_trace(go.Bar(
        name='Non-thinking',
        x=diff_df["Difficulty"],
        y=diff_df["Non-thinking"],
        marker_color='#FF6B6B',
        text=diff_df["Non-thinking"].apply(lambda x: f'{x:.1%}'),
        textposition='auto',
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Non-thinking Accuracy: %{y:.1%}<br>" +
            "Questions: %{customdata}<br>" +
            "<extra></extra>"
        ),
        customdata=diff_df["Count"]
    ))
    
    # Add bars for thinking accuracy
    fig.add_trace(go.Bar(
        name='Thinking',
        x=diff_df["Difficulty"],
        y=diff_df["Thinking"],
        marker_color='#4ECDC4',
        text=diff_df["Thinking"].apply(lambda x: f'{x:.1%}'),
        textposition='auto',
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Thinking Accuracy: %{y:.1%}<br>" +
            "Questions: %{customdata}<br>" +
            "<extra></extra>"
        ),
        customdata=diff_df["Count"]
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Accuracy by Question Difficulty (Model Confidence)<br>" +
                   f"<sub>Based on {len(paired_df)} paired questions</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis=dict(
            title="Difficulty Level (Based on Model Confidence)",
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Accuracy",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            tickformat=".0%",
            range=[0, 1]
        ),
        barmode='group',
        height=600,
        width=900,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add annotations for question counts
    for i, row in diff_df.iterrows():
        fig.add_annotation(
            text=f"n={row['Count']}",
            x=row["Difficulty"],
            y=-0.05,
            xref="x",
            yref="y",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Save the figure
    fig.write_html(output_path)
    print(f"Interactive difficulty plot saved to {output_path}")


def create_confidence_calibration_plot(
    thinking_df: pd.DataFrame,
    output_path: str = "results/analysis/confidence_calibration_interactive.html",
    n_bins: int = 20,
    mode: str = "bins"  # "bins" or "subjects"
) -> None:
    """
    Create an interactive scatter plot showing model confidence vs actual accuracy.
    
    Args:
        thinking_df: DataFrame with model predictions and confidence scores
        output_path: Where to save the HTML file
        n_bins: Number of bins for confidence grouping (if mode="bins")
        mode: "bins" for confidence bins, "subjects" for per-subject analysis
    """
    # Ensure we have the required columns
    if "prob_generated" not in thinking_df.columns:
        print("Warning: prob_generated column not found")
        return
    
    if mode == "bins":
        # Create confidence bins
        thinking_df = thinking_df.copy()
        thinking_df["confidence_bin"] = pd.cut(
            thinking_df["prob_generated"],
            bins=n_bins,
            labels=False,
            include_lowest=True
        )
        
        # Calculate statistics for each bin
        bin_data = []
        for bin_idx in range(n_bins):
            subset = thinking_df[thinking_df["confidence_bin"] == bin_idx]
            if len(subset) > 0:
                avg_confidence = subset["prob_generated"].mean()
                actual_accuracy = subset["is_correct"].mean()
                bin_data.append({
                    "avg_confidence": avg_confidence,
                    "actual_accuracy": actual_accuracy,
                    "count": len(subset),
                    "bin": bin_idx,
                    "confidence_range": f"{subset['prob_generated'].min():.2f}-{subset['prob_generated'].max():.2f}"
                })
        
        plot_df = pd.DataFrame(bin_data)
        hover_template = (
            "<b>Confidence Range: %{customdata[0]}</b><br>" +
            "Avg Confidence: %{x:.1%}<br>" +
            "Actual Accuracy: %{y:.1%}<br>" +
            "Questions: %{customdata[1]}<br>" +
            "Calibration Error: %{customdata[2]:.1%}<br>" +
            "<extra></extra>"
        )
        
    else:  # mode == "subjects"
        if "subject" not in thinking_df.columns:
            print("No subject column found for subject-level analysis")
            return
        
        # Calculate per-subject statistics
        subject_data = []
        for subject in thinking_df["subject"].unique():
            subset = thinking_df[thinking_df["subject"] == subject]
            if len(subset) >= 10:  # Minimum questions per subject
                avg_confidence = subset["prob_generated"].mean()
                actual_accuracy = subset["is_correct"].mean()
                subject_data.append({
                    "subject": subject,
                    "subject_display": subject.replace("_", " ").title(),
                    "avg_confidence": avg_confidence,
                    "actual_accuracy": actual_accuracy,
                    "count": len(subset)
                })
        
        plot_df = pd.DataFrame(subject_data)
        hover_template = (
            "<b>%{text}</b><br>" +
            "Avg Confidence: %{x:.1%}<br>" +
            "Actual Accuracy: %{y:.1%}<br>" +
            "Questions: %{customdata[0]}<br>" +
            "Calibration Error: %{customdata[1]:.1%}<br>" +
            "<extra></extra>"
        )
    
    # Calculate calibration errors
    plot_df["calibration_error"] = plot_df["avg_confidence"] - plot_df["actual_accuracy"]
    plot_df["abs_calibration_error"] = plot_df["calibration_error"].abs()
    
    # Determine color based on calibration
    def get_color(error):
        if abs(error) < 0.05:
            return "#2ecc71"  # Well calibrated (green)
        elif abs(error) < 0.1:
            return "#f39c12"  # Slightly miscalibrated (orange)
        elif error > 0:
            return "#e74c3c"  # Overconfident (red)
        else:
            return "#3498db"  # Underconfident (blue)
    
    plot_df["color"] = plot_df["calibration_error"].apply(get_color)
    
    # Create the figure
    fig = go.Figure()
    
    # Add the scatter plot
    if mode == "bins":
        customdata = np.column_stack((
            plot_df["confidence_range"],
            plot_df["count"],
            plot_df["calibration_error"]
        ))
        text = None
    else:
        customdata = np.column_stack((
            plot_df["count"],
            plot_df["calibration_error"]
        ))
        text = plot_df["subject_display"]
    
    fig.add_trace(go.Scatter(
        x=plot_df["avg_confidence"],
        y=plot_df["actual_accuracy"],
        mode='markers+text' if mode == "subjects" else 'markers',
        marker=dict(
            size=plot_df["count"].apply(lambda x: min(30, 5 + x * 0.02)),
            color=plot_df["color"],
            line=dict(width=1, color='black')
        ),
        text=text,
        textposition="top center" if mode == "subjects" else None,
        textfont=dict(size=8) if mode == "subjects" else None,
        hovertemplate=hover_template,
        customdata=customdata,
        showlegend=False
    ))
    
    # Set axis ranges based on data
    x_min = plot_df["avg_confidence"].min() - 0.05
    x_max = plot_df["avg_confidence"].max() + 0.05
    y_min = plot_df["actual_accuracy"].min() - 0.05
    y_max = plot_df["actual_accuracy"].max() + 0.05
    
    # Calculate and display calibration metrics
    mean_confidence = plot_df["avg_confidence"].mean()
    mean_accuracy = plot_df["actual_accuracy"].mean()
    ece = (plot_df["abs_calibration_error"] * plot_df["count"]).sum() / plot_df["count"].sum()
    
    title_text = "Model Confidence Calibration" if mode == "bins" else "Confidence Calibration by Subject"
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{title_text}<br>" +
                   f"<sub>ECE: {ece:.3f} | Avg Confidence: {mean_confidence:.1%} | Avg Accuracy: {mean_accuracy:.1%}</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis=dict(
            title="Model Confidence (prob_generated)",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            tickformat=".0%",
            range=[x_min, x_max],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            title="Actual Accuracy",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            tickformat=".0%",
            range=[y_min, y_max],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        height=700,
        width=1000,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Black",
            borderwidth=1
        )
    )
    
    # Add note about calibration
    fig.add_annotation(
        text="Point size indicates number of questions",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=11),
        align="center"
    )
    
    # Add legend for colors
    legend_text = (
        "<b>Calibration:</b><br>" +
        "🟢 Well calibrated (±5%)<br>" +
        "🟠 Slightly off (±5-10%)<br>" +
        "🔴 Overconfident (>10%)<br>" +
        "🔵 Underconfident (<-10%)"
    )
    
    fig.add_annotation(
        text=legend_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10),
        align="left"
    )
    
    # Save the figure
    fig.write_html(output_path)
    print(f"Confidence calibration plot saved to {output_path}")