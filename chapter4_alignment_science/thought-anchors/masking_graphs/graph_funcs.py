"""
Graph and network analysis functions for upper triangular matrices.

This module analyzes causal connections and dependencies in chain-of-thought
reasoning by treating sentence relationships as a directed graph/network.
"""

import numpy as np
import networkx as nx
from scipy import stats, signal
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Any, Optional, Tuple, List
import warnings

def analyze_thought_graph(
    matrix: np.ndarray,
    threshold_percentile: float = 25,
    distance_decay: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive analysis of chain-of-thought causal structure.
    
    Args:
        matrix: Upper triangular matrix (lower triangle should be NaN)
        threshold_percentile: Percentile for edge thresholding in graph construction
        distance_decay: Whether to apply distance-based weighting
        
    Returns:
        Dictionary with various graph and statistical metrics
    """
    results = {}
    
    # Basic validation
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    
    # Extract upper triangle values (excluding diagonal)
    upper_mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    upper_values = matrix[upper_mask]
    upper_values_clean = upper_values[~np.isnan(upper_values)]
    
    if len(upper_values_clean) == 0:
        return {"error": "No valid values in upper triangle"}
    
    # 1. Basic Statistics
    results.update(_compute_basic_stats(upper_values_clean))
    
    # 2. Distance-aware Statistics
    results.update(_compute_distance_stats(matrix, upper_mask))
    
    # 3. Graph/Network Analysis
    results.update(_compute_graph_metrics(matrix, threshold_percentile))
    
    # 4. Information Flow Metrics
    results.update(_compute_information_flow(matrix))
    
    # 5. Sequential Pattern Analysis
    results.update(_compute_sequential_patterns(matrix))
    
    # 6. Locality and Range Dependencies
    results.update(_compute_locality_metrics(matrix))
    
    # 7. Hierarchical Structure
    results.update(_compute_hierarchical_metrics(matrix))
    
    return results


def _compute_basic_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute basic statistical measures."""
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "variance": float(np.var(values)),
        "skewness": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values)),
        "entropy": float(stats.entropy(values + 1e-10)),  # Add small constant to avoid log(0)
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "cv": float(np.std(values) / (np.mean(values) + 1e-10)),  # Coefficient of variation
        "gini": float(_gini_coefficient(values)),
    }


def _compute_distance_stats(matrix: np.ndarray, upper_mask: np.ndarray) -> Dict[str, float]:
    """Compute statistics that account for distance from diagonal."""
    n = matrix.shape[0]
    results = {}
    
    # Group values by distance from diagonal
    distance_means = []
    distance_stds = []
    max_distance = n - 1
    
    for k in range(1, min(n, 20)):  # Analyze first 20 diagonals
        diag_values = np.diag(matrix, k)
        diag_values_clean = diag_values[~np.isnan(diag_values)]
        if len(diag_values_clean) > 0:
            distance_means.append(np.mean(diag_values_clean))
            distance_stds.append(np.std(diag_values_clean))
    
    if distance_means:
        # Decay rate (how quickly values decrease with distance)
        distances = np.arange(1, len(distance_means) + 1)
        if len(distances) > 1:
            slope, intercept, r_value, _, _ = stats.linregress(distances, distance_means)
            results["distance_decay_rate"] = float(slope)
            results["distance_decay_r2"] = float(r_value ** 2)
        
        results["near_diagonal_mean"] = float(np.mean(distance_means[:3])) if len(distance_means) >= 3 else float(distance_means[0])
        results["far_diagonal_mean"] = float(np.mean(distance_means[-3:])) if len(distance_means) >= 6 else float(distance_means[-1])
        results["diagonal_contrast_ratio"] = results["near_diagonal_mean"] / (results["far_diagonal_mean"] + 1e-10)
    
    # Weighted average by inverse distance
    weighted_sum = 0
    weight_sum = 0
    for i in range(n):
        for j in range(i+1, n):
            if not np.isnan(matrix[i, j]):
                weight = 1.0 / (j - i)
                weighted_sum += matrix[i, j] * weight
                weight_sum += weight
    
    if weight_sum > 0:
        results["distance_weighted_mean"] = float(weighted_sum / weight_sum)
    
    return results


def _compute_graph_metrics(matrix: np.ndarray, threshold_percentile: float) -> Dict[str, float]:
    """Compute graph-theoretic metrics using NetworkX."""
    n = matrix.shape[0]
    results = {}
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Determine threshold for edge inclusion
    upper_values = matrix[np.triu(np.ones_like(matrix, dtype=bool), k=1)]
    upper_values_clean = upper_values[~np.isnan(upper_values)]
    
    if len(upper_values_clean) == 0:
        return results
    
    threshold = np.percentile(upper_values_clean, threshold_percentile)
    
    # Add edges above threshold
    edge_count = 0
    total_weight = 0
    for i in range(n):
        for j in range(i+1, n):
            if not np.isnan(matrix[i, j]) and matrix[i, j] > threshold:
                G.add_edge(i, j, weight=matrix[i, j])
                edge_count += 1
                total_weight += matrix[i, j]
    
    if edge_count == 0:
        return results
    
    # Basic graph properties
    results["num_edges"] = edge_count
    results["edge_density"] = float(edge_count / (n * (n - 1) / 2))
    results["avg_edge_weight"] = float(total_weight / edge_count) if edge_count > 0 else 0
    
    # Connectivity
    results["num_connected_components"] = nx.number_weakly_connected_components(G)
    results["largest_component_size"] = len(max(nx.weakly_connected_components(G), key=len)) if G.nodes() else 0
    
    # Centrality measures
    if G.nodes():
        # Degree centrality
        in_centrality = nx.in_degree_centrality(G)
        out_centrality = nx.out_degree_centrality(G)
        results["avg_in_degree_centrality"] = float(np.mean(list(in_centrality.values())))
        results["avg_out_degree_centrality"] = float(np.mean(list(out_centrality.values())))
        results["max_in_degree_centrality"] = float(max(in_centrality.values())) if in_centrality else 0
        results["max_out_degree_centrality"] = float(max(out_centrality.values())) if out_centrality else 0
        
        # Betweenness centrality (how often nodes appear on shortest paths)
        try:
            betweenness = nx.betweenness_centrality(G, weight='weight')
            results["avg_betweenness_centrality"] = float(np.mean(list(betweenness.values())))
            results["max_betweenness_centrality"] = float(max(betweenness.values()))
        except:
            pass
        
        # PageRank (importance based on connections)
        try:
            pagerank = nx.pagerank(G, weight='weight')
            results["avg_pagerank"] = float(np.mean(list(pagerank.values())))
            results["max_pagerank"] = float(max(pagerank.values()))
            results["pagerank_entropy"] = float(stats.entropy(list(pagerank.values()) + [1e-10]))
        except:
            pass
        
        # Clustering coefficient
        try:
            clustering = nx.clustering(G.to_undirected(), weight='weight')
            results["avg_clustering_coefficient"] = float(np.mean(list(clustering.values())))
        except:
            pass
        
        # Path statistics
        try:
            if nx.is_weakly_connected(G):
                # Average shortest path length
                avg_path_length = nx.average_shortest_path_length(G, weight='weight')
                results["avg_shortest_path_length"] = float(avg_path_length)
                
                # Diameter (longest shortest path)
                diameter = nx.diameter(G.to_undirected())
                results["graph_diameter"] = diameter
        except:
            pass
    
    # Transitivity (probability of triangles)
    try:
        results["transitivity"] = float(nx.transitivity(G))
    except:
        pass
    
    # Assortativity (do similar nodes connect?)
    try:
        results["degree_assortativity"] = float(nx.degree_assortativity_coefficient(G, weight='weight'))
    except:
        pass
    
    return results


def _compute_information_flow(matrix: np.ndarray) -> Dict[str, float]:
    """Compute metrics related to information flow through the network."""
    n = matrix.shape[0]
    results = {}
    
    # Forward flow strength (sum of outgoing connections per node)
    forward_flow = []
    for i in range(n-1):
        row_values = matrix[i, i+1:]
        row_clean = row_values[~np.isnan(row_values)]
        if len(row_clean) > 0:
            forward_flow.append(np.sum(row_clean))
    
    if forward_flow:
        results["avg_forward_flow"] = float(np.mean(forward_flow))
        results["max_forward_flow"] = float(np.max(forward_flow))
        results["forward_flow_variance"] = float(np.var(forward_flow))
        
        # Identify bottlenecks (nodes with low forward flow)
        if len(forward_flow) > 5:
            bottleneck_threshold = np.percentile(forward_flow, 20)
            results["num_bottlenecks"] = int(np.sum(np.array(forward_flow) < bottleneck_threshold))
    
    # Cumulative influence (how much each position influences future positions)
    cumulative_influence = []
    for i in range(n):
        influence_values = matrix[i, :]
        influence_clean = influence_values[~np.isnan(influence_values)]
        if len(influence_clean) > 0:
            cumulative_influence.append(np.sum(influence_clean))
    
    if cumulative_influence:
        results["avg_cumulative_influence"] = float(np.mean(cumulative_influence))
        results["influence_peak_position"] = int(np.argmax(cumulative_influence))
        results["influence_concentration"] = float(np.max(cumulative_influence) / (np.sum(cumulative_influence) + 1e-10))
    
    # Information diffusion rate (how quickly information spreads)
    diffusion_rates = []
    for i in range(min(n-1, 10)):  # Check first 10 positions
        local_values = []
        for k in range(1, min(6, n-i)):  # Look at next 5 positions
            if i+k < n and not np.isnan(matrix[i, i+k]):
                local_values.append(matrix[i, i+k] / k)  # Normalize by distance
        if local_values:
            diffusion_rates.append(np.mean(local_values))
    
    if diffusion_rates:
        results["avg_diffusion_rate"] = float(np.mean(diffusion_rates))
        results["diffusion_rate_std"] = float(np.std(diffusion_rates))
    
    return results


def _compute_sequential_patterns(matrix: np.ndarray) -> Dict[str, float]:
    """Analyze sequential and temporal patterns in the thought chain."""
    n = matrix.shape[0]
    results = {}
    
    # Local coherence (strength of adjacent connections)
    diagonal_1 = np.diag(matrix, k=1)
    diagonal_1_clean = diagonal_1[~np.isnan(diagonal_1)]
    
    if len(diagonal_1_clean) > 0:
        results["local_coherence_mean"] = float(np.mean(diagonal_1_clean))
        results["local_coherence_std"] = float(np.std(diagonal_1_clean))
        results["local_coherence_cv"] = float(np.std(diagonal_1_clean) / (np.mean(diagonal_1_clean) + 1e-10))
        
        # Detect breaks in local coherence
        if len(diagonal_1_clean) > 3:
            median_coherence = np.median(diagonal_1_clean)
            breaks = np.sum(diagonal_1_clean < 0.5 * median_coherence)
            results["num_coherence_breaks"] = int(breaks)
    
    # Skip connections (strength of non-adjacent connections)
    skip_strengths = []
    for k in range(2, min(6, n)):
        diag_k = np.diag(matrix, k)
        diag_k_clean = diag_k[~np.isnan(diag_k)]
        if len(diag_k_clean) > 0:
            skip_strengths.append(np.mean(diag_k_clean))
    
    if skip_strengths:
        results["avg_skip_connection_strength"] = float(np.mean(skip_strengths))
        results["skip_connection_decay"] = float((skip_strengths[0] - skip_strengths[-1]) / (skip_strengths[0] + 1e-10)) if len(skip_strengths) > 1 else 0
    
    # Periodicity detection (using autocorrelation on row sums)
    row_sums = []
    for i in range(n):
        row_values = matrix[i, :]
        row_clean = row_values[~np.isnan(row_values)]
        if len(row_clean) > 0:
            row_sums.append(np.sum(row_clean))
    
    if len(row_sums) > 10:
        # Compute autocorrelation
        row_sums_array = np.array(row_sums)
        autocorr = np.correlate(row_sums_array - np.mean(row_sums_array), 
                               row_sums_array - np.mean(row_sums_array), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find peaks in autocorrelation
        if len(autocorr) > 3:
            peaks, properties = signal.find_peaks(autocorr[1:min(len(autocorr), 20)], height=0.3)
            if len(peaks) > 0:
                results["periodicity_strength"] = float(np.max(properties['peak_heights']))
                results["primary_period"] = int(peaks[0] + 1)
            else:
                results["periodicity_strength"] = 0.0
                results["primary_period"] = 0
    
    # Momentum (increasing or decreasing trend in connections)
    if len(diagonal_1_clean) > 5:
        x = np.arange(len(diagonal_1_clean))
        slope, _, r_value, _, _ = stats.linregress(x, diagonal_1_clean)
        results["sequential_momentum"] = float(slope)
        results["momentum_r2"] = float(r_value ** 2)
    
    return results


def _compute_locality_metrics(matrix: np.ndarray) -> Dict[str, float]:
    """Compute metrics related to local vs global dependencies."""
    n = matrix.shape[0]
    results = {}
    
    # Compute local vs global strength ratio
    local_sum = 0
    local_count = 0
    global_sum = 0
    global_count = 0
    
    local_window = 3  # Define local as within 3 steps
    
    for i in range(n):
        for j in range(i+1, n):
            if not np.isnan(matrix[i, j]):
                if j - i <= local_window:
                    local_sum += matrix[i, j]
                    local_count += 1
                else:
                    global_sum += matrix[i, j]
                    global_count += 1
    
    if local_count > 0:
        results["local_strength"] = float(local_sum / local_count)
    if global_count > 0:
        results["global_strength"] = float(global_sum / global_count)
    if local_count > 0 and global_count > 0:
        results["locality_ratio"] = float((local_sum / local_count) / (global_sum / global_count + 1e-10))
    
    # Receptive field analysis (how far back each position looks)
    receptive_fields = []
    for j in range(1, n):
        col_values = matrix[:j, j]
        col_clean = col_values[~np.isnan(col_values)]
        if len(col_clean) > 0:
            # Weighted average distance
            weights = col_clean
            distances = np.arange(j, 0, -1)[-len(col_clean):]
            weighted_distance = np.sum(weights * distances) / (np.sum(weights) + 1e-10)
            receptive_fields.append(weighted_distance)
    
    if receptive_fields:
        results["avg_receptive_field"] = float(np.mean(receptive_fields))
        results["max_receptive_field"] = float(np.max(receptive_fields))
        results["receptive_field_growth"] = float((receptive_fields[-1] - receptive_fields[0]) / (len(receptive_fields) + 1e-10)) if len(receptive_fields) > 1 else 0
    
    # Long-range dependency strength
    long_range_values = []
    for i in range(n):
        for j in range(i+5, n):  # Consider connections 5+ steps away as long-range
            if not np.isnan(matrix[i, j]):
                long_range_values.append(matrix[i, j])
    
    if long_range_values:
        results["long_range_mean"] = float(np.mean(long_range_values))
        results["long_range_max"] = float(np.max(long_range_values))
        results["long_range_prevalence"] = float(len(long_range_values) / (n * (n-1) / 2))
    
    return results


def _compute_hierarchical_metrics(matrix: np.ndarray) -> Dict[str, float]:
    """Compute metrics related to hierarchical structure."""
    n = matrix.shape[0]
    results = {}
    
    # Compute influence hierarchy (nodes that influence many others)
    out_influences = []
    in_influences = []
    
    for i in range(n):
        # Outgoing influence
        row_values = matrix[i, :]
        row_clean = row_values[~np.isnan(row_values)]
        if len(row_clean) > 0:
            out_influences.append(np.sum(row_clean))
        else:
            out_influences.append(0)
        
        # Incoming influence
        col_values = matrix[:, i]
        col_clean = col_values[~np.isnan(col_values)]
        if len(col_clean) > 0:
            in_influences.append(np.sum(col_clean))
        else:
            in_influences.append(0)
    
    if out_influences:
        # Hierarchy based on influence imbalance
        influence_imbalance = np.array(out_influences) - np.array(in_influences)
        results["hierarchy_strength"] = float(np.std(influence_imbalance))
        results["top_influencer_position"] = int(np.argmax(out_influences))
        results["top_influenced_position"] = int(np.argmax(in_influences))
        
        # Identify hubs (high out-influence)
        if len(out_influences) > 5:
            hub_threshold = np.percentile(out_influences, 80)
            results["num_hubs"] = int(np.sum(np.array(out_influences) > hub_threshold))
    
    # Cascade potential (how much early nodes influence later ones)
    cascade_strengths = []
    for i in range(min(5, n//3)):  # Check first third of nodes
        future_influence = matrix[i, i+1:]
        future_clean = future_influence[~np.isnan(future_influence)]
        if len(future_clean) > 0:
            cascade_strengths.append(np.sum(future_clean))
    
    if cascade_strengths:
        results["cascade_potential"] = float(np.mean(cascade_strengths))
        results["cascade_variance"] = float(np.var(cascade_strengths))
    
    # Modularity (tendency to form clusters)
    # Simplified modularity: variance in local density
    window_size = max(3, n // 5)
    local_densities = []
    
    for i in range(0, n - window_size + 1):
        window = matrix[i:i+window_size, i:i+window_size]
        window_upper = window[np.triu_indices(window_size, k=1)]
        window_clean = window_upper[~np.isnan(window_upper)]
        if len(window_clean) > 0:
            local_densities.append(np.mean(window_clean))
    
    if len(local_densities) > 1:
        results["modularity_variance"] = float(np.var(local_densities))
        results["modularity_coefficient"] = float(np.std(local_densities) / (np.mean(local_densities) + 1e-10))
    
    return results


def _gini_coefficient(values: np.ndarray) -> float:
    """Calculate Gini coefficient for inequality measurement."""
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    return float(gini)


def get_thought_graph_features(matrix: np.ndarray) -> Dict[str, float]:
    """
    Main entry point for getting all graph features.
    
    This is the single function to call to get all graph variables.
    
    Args:
        matrix: Upper triangular NumPy array (lower triangle should be NaN)
        
    Returns:
        Dictionary with all computed features
    """
    try:
        return analyze_thought_graph(matrix, threshold_percentile=25, distance_decay=True)
    except Exception as e:
        return {"error": str(e)}


# Convenience function for quick summary
def get_key_metrics(matrix: np.ndarray) -> Dict[str, float]:
    """
    Get a smaller set of the most important metrics.
    
    Args:
        matrix: Upper triangular NumPy array
        
    Returns:
        Dictionary with key metrics only
    """
    full_metrics = get_thought_graph_features(matrix)
    
    # Select most interpretable/important metrics
    key_metric_names = [
        "mean", "std", "entropy", "gini",
        "distance_decay_rate", "diagonal_contrast_ratio",
        "edge_density", "avg_clustering_coefficient",
        "avg_pagerank", "transitivity",
        "avg_forward_flow", "influence_concentration",
        "local_coherence_mean", "num_coherence_breaks",
        "locality_ratio", "avg_receptive_field",
        "hierarchy_strength", "cascade_potential",
        "modularity_coefficient"
    ]
    
    return {k: v for k, v in full_metrics.items() if k in key_metric_names}


if __name__ == "__main__":
    # Example usage and testing
    np.random.seed(42)
    
    # Create a sample upper triangular matrix
    n = 20
    matrix = np.full((n, n), np.nan)
    
    # Fill upper triangle with values that decay with distance
    for i in range(n):
        for j in range(i+1, n):
            distance = j - i
            # Higher values near diagonal, with some noise
            base_value = 1.0 / (distance ** 0.5)
            noise = np.random.normal(0, 0.1)
            matrix[i, j] = max(0, base_value + noise)
    
    # Get all features
    print("Computing all graph features...")
    all_features = get_thought_graph_features(matrix)
    
    print(f"\nTotal features computed: {len(all_features)}")
    print("\nSample of computed features:")
    for key in list(all_features.keys())[:20]:
        print(f"  {key}: {all_features[key]:.4f}")
    
    # Get key metrics only
    print("\n\nKey metrics summary:")
    key_metrics = get_key_metrics(matrix)
    for key, value in key_metrics.items():
        print(f"  {key}: {value:.4f}")