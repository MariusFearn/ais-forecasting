"""Route clustering using existing DTW implementation from src/models/clustering.py."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
import time

from ..models.clustering import compute_dtw_distance_matrix, cluster_routes
from ..utils.logging_setup import get_logger, log_performance_metrics

logger = get_logger(__name__)

def cluster_shipping_routes(
    journey_data: pd.DataFrame,
    config: Dict,
    max_routes: Optional[int] = None
) -> pd.DataFrame:
    """
    Cluster shipping routes using existing DTW implementation.
    
    Leverages:
    - compute_dtw_distance_matrix() from src/models/clustering.py
    - cluster_routes() from src/models/clustering.py
    
    Args:
        journey_data: DataFrame with h3_sequence column
        config: Configuration dictionary with clustering parameters
        max_routes: Optional override for maximum routes to process
        
    Returns:
        DataFrame with added route_cluster column
        
    Raises:
        ValueError: If input data is invalid
    """
    logger.info("Clustering shipping routes using DTW")
    start_time = time.time()
    
    if journey_data.empty:
        logger.warning("No journey data provided for route clustering")
        return journey_data
    
    if 'h3_sequence' not in journey_data.columns:
        raise ValueError("Journey data must contain 'h3_sequence' column")
    
    # Extract H3 sequences for clustering
    h3_sequences = journey_data['h3_sequence'].tolist()
    
    # Filter sequences by length
    sequence_limits = config.get('sequence_limits', {})
    min_length = sequence_limits.get('min_length', 5)
    max_length = sequence_limits.get('max_length', 200)
    
    valid_sequences = []
    valid_indices = []
    
    for idx, seq in enumerate(h3_sequences):
        if seq and min_length <= len(seq) <= max_length:
            valid_sequences.append(seq)
            valid_indices.append(idx)
    
    logger.info(f"Route filtering: {len(valid_sequences):,} valid sequences "
                f"out of {len(h3_sequences):,} total")
    
    if not valid_sequences:
        logger.warning("No valid sequences for route clustering")
        journey_data['route_cluster'] = -1
        return journey_data
    
    # Limit routes for DTW performance
    max_dtw_routes = max_routes or config.get('max_routes_for_dtw', 150)
    
    if len(valid_sequences) > max_dtw_routes:
        logger.info(f"Limiting to {max_dtw_routes} routes for DTW performance")
        
        # Sample routes for diverse coverage
        step = len(valid_sequences) // max_dtw_routes
        sampled_indices = list(range(0, len(valid_sequences), step))[:max_dtw_routes]
        dtw_sequences = [valid_sequences[i] for i in sampled_indices]
        dtw_journey_indices = [valid_indices[i] for i in sampled_indices]
    else:
        dtw_sequences = valid_sequences
        dtw_journey_indices = valid_indices
    
    logger.info(f"Computing DTW distance matrix for {len(dtw_sequences):,} routes")
    
    try:
        # Use existing DTW implementation
        distance_matrix = compute_dtw_distance_matrix(dtw_sequences)
        
        # Use existing clustering implementation
        cluster_result = cluster_routes(distance_matrix, config)
        
        # Extract cluster labels
        if isinstance(cluster_result, tuple):
            route_clusters = cluster_result[0]
        else:
            route_clusters = cluster_result
        
        # Convert to list if numpy array
        if hasattr(route_clusters, 'tolist'):
            route_clusters = route_clusters.tolist()
        else:
            route_clusters = list(route_clusters)
        
        # Assign clusters back to journey data
        full_clusters = [-1] * len(journey_data)  # Default to noise
        
        for i, journey_idx in enumerate(dtw_journey_indices):
            if i < len(route_clusters) and journey_idx < len(full_clusters):
                full_clusters[journey_idx] = route_clusters[i]
        
        journey_data = journey_data.copy()
        journey_data['route_cluster'] = full_clusters
        
        # Log clustering results
        n_clusters = len(set(route_clusters)) - (1 if -1 in route_clusters else 0)
        n_noise = route_clusters.count(-1)
        
        duration = time.time() - start_time
        log_performance_metrics(
            logger, 
            "Route Clustering", 
            duration, 
            len(dtw_sequences)
        )
        
        logger.info(f"Route clustering results:")
        logger.info(f"   Clusters found: {n_clusters}")
        logger.info(f"   Routes in clusters: {len(route_clusters) - n_noise}")
        logger.info(f"   Outlier routes: {n_noise}")
        
        return journey_data
        
    except Exception as e:
        logger.error(f"Route clustering failed: {e}")
        journey_data = journey_data.copy()
        journey_data['route_cluster'] = -1  # All noise
        return journey_data

def optimize_route_selection(
    sequences: List[List[str]],
    max_routes: int,
    strategy: str = "diverse"
) -> Tuple[List[List[str]], List[int]]:
    """
    Select representative routes for DTW computation.
    
    Args:
        sequences: List of H3 sequences
        max_routes: Maximum number of routes to select
        strategy: Selection strategy ("diverse", "random", "representative")
        
    Returns:
        Tuple of (selected_sequences, selected_indices)
    """
    if len(sequences) <= max_routes:
        return sequences, list(range(len(sequences)))
    
    if strategy == "diverse":
        # Sample evenly across the dataset
        step = len(sequences) // max_routes
        selected_indices = list(range(0, len(sequences), step))[:max_routes]
    elif strategy == "random":
        # Random sampling
        selected_indices = np.random.choice(len(sequences), max_routes, replace=False).tolist()
    else:  # "representative"
        # Select based on sequence length distribution
        lengths = [len(seq) for seq in sequences]
        length_percentiles = np.percentile(lengths, np.linspace(0, 100, max_routes))
        selected_indices = []
        
        for target_length in length_percentiles:
            # Find sequence closest to target length
            closest_idx = min(range(len(lengths)), 
                            key=lambda i: abs(lengths[i] - target_length))
            if closest_idx not in selected_indices:
                selected_indices.append(closest_idx)
        
        # Fill remaining slots with diverse sampling
        while len(selected_indices) < max_routes:
            remaining_indices = [i for i in range(len(sequences)) 
                               if i not in selected_indices]
            if remaining_indices:
                selected_indices.append(remaining_indices[0])
            else:
                break
    
    selected_sequences = [sequences[i] for i in selected_indices]
    return selected_sequences, selected_indices

def analyze_route_clusters(journey_data: pd.DataFrame) -> Dict:
    """
    Analyze route clustering results.
    
    Args:
        journey_data: DataFrame with route_cluster column
        
    Returns:
        Dictionary with clustering analysis
    """
    if 'route_cluster' not in journey_data.columns:
        return {}
    
    clusters = journey_data['route_cluster']
    n_clusters = len(set(clusters)) - (1 if -1 in clusters.values else 0)
    n_noise = (clusters == -1).sum()
    
    analysis = {
        'total_routes': len(journey_data),
        'clusters_found': n_clusters,
        'routes_in_clusters': len(journey_data) - n_noise,
        'outlier_routes': n_noise,
        'clustering_rate': (len(journey_data) - n_noise) / len(journey_data) if len(journey_data) > 0 else 0
    }
    
    # Cluster size distribution
    if n_clusters > 0:
        cluster_sizes = journey_data[journey_data['route_cluster'] != -1]['route_cluster'].value_counts()
        analysis['largest_cluster_size'] = cluster_sizes.max()
        analysis['average_cluster_size'] = cluster_sizes.mean()
        analysis['cluster_size_std'] = cluster_sizes.std()
    
    return analysis
