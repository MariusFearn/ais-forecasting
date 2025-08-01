import torch
import numpy as np
import pandas as pd
import logging
from typing import Union, Callable, Dict, Any
from sklearn.metrics import silhouette_score
from collections import Counter


def mae(y_pred: Union[torch.Tensor, np.ndarray], y_true: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        
    Returns:
        float: MAE value
    """
    if isinstance(y_pred, torch.Tensor):
        return (y_pred - y_true).abs().mean().item()
    else:
        return np.mean(np.abs(y_pred - y_true))


def rmse(y_pred: Union[torch.Tensor, np.ndarray], y_true: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        
    Returns:
        float: RMSE value
    """
    if isinstance(y_pred, torch.Tensor):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
    else:
        return np.sqrt(np.mean((y_pred - y_true) ** 2))


def smape(y_pred: Union[torch.Tensor, np.ndarray], y_true: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        
    Returns:
        float: SMAPE value (0-100)
    """
    if isinstance(y_pred, torch.Tensor):
        return 100 * torch.mean(2 * torch.abs(y_pred - y_true) / (torch.abs(y_pred) + torch.abs(y_true) + 1e-8)).item()
    else:
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))


def mape(y_pred: Union[torch.Tensor, np.ndarray], y_true: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        
    Returns:
        float: MAPE value (0-100)
    """
    if isinstance(y_pred, torch.Tensor):
        return 100 * torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))).item()
    else:
        return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))


def quantile_loss(y_pred: Union[torch.Tensor, np.ndarray], y_true: Union[torch.Tensor, np.ndarray], 
                 quantiles: list = [0.1, 0.5, 0.9]) -> Dict[str, float]:
    """
    Calculate quantile loss for each quantile.
    
    Args:
        y_pred: Predicted values (shape: batch x quantiles)
        y_true: Ground truth values (shape: batch)
        quantiles: List of quantiles
        
    Returns:
        Dict[str, float]: Dictionary with quantile loss for each quantile
    """
    results = {}
    
    if isinstance(y_pred, torch.Tensor):
        for i, q in enumerate(quantiles):
            errors = y_true - y_pred[:, i]
            results[f"q{int(q*100)}"] = torch.mean((q - (errors < 0).float()) * errors).item()
    else:
        for i, q in enumerate(quantiles):
            errors = y_true - y_pred[:, i]
            results[f"q{int(q*100)}"] = np.mean((q - (errors < 0).astype(float)) * errors)
    
    return results


class MetricCollection:
    """Collection of metrics to compute on model outputs."""
    
    def __init__(self, metrics: Dict[str, Callable]):
        """
        Initialize with a dictionary of metric functions.
        
        Args:
            metrics: Dictionary mapping metric names to metric functions
        """
        self.metrics = metrics
    
    def __call__(self, y_pred: Union[torch.Tensor, np.ndarray], 
                y_true: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """
        Compute all metrics on the given predictions and targets.
        
        Args:
            y_pred: Predicted values
            y_true: Ground truth values
            
        Returns:
            Dict[str, float]: Dictionary mapping metric names to computed values
        """
        results = {}
        for name, metric_fn in self.metrics.items():
            results[name] = metric_fn(y_pred, y_true)
        
        return results


# Default metric collection for time series forecasting
default_metrics = MetricCollection({
    'mae': mae,
    'rmse': rmse,
    'smape': smape,
    'mape': mape,
})

# ===== LANE DISCOVERY VALIDATION METRICS =====

def calculate_clustering_metrics(distance_matrix: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
    """
    Calculate clustering quality metrics for lane discovery.

    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix used for clustering.
        cluster_labels (np.ndarray): Cluster labels from DBSCAN.

    Returns:
        Dict[str, Any]: Dictionary of clustering metrics.
    """
    metrics = {}
    
    # Basic cluster statistics
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_outliers = list(cluster_labels).count(-1)
    n_total = len(cluster_labels)
    
    metrics['n_clusters'] = n_clusters
    metrics['n_outliers'] = n_outliers
    metrics['n_total_journeys'] = n_total
    metrics['outlier_percentage'] = (n_outliers / n_total) * 100 if n_total > 0 else 0
    
    # Cluster size distribution
    cluster_counts = Counter(cluster_labels)
    if -1 in cluster_counts:
        del cluster_counts[-1]  # Remove outliers from size analysis
    
    if cluster_counts:
        cluster_sizes = list(cluster_counts.values())
        metrics['avg_cluster_size'] = np.mean(cluster_sizes)
        metrics['median_cluster_size'] = np.median(cluster_sizes)
        metrics['min_cluster_size'] = min(cluster_sizes)
        metrics['max_cluster_size'] = max(cluster_sizes)
        metrics['cluster_size_std'] = np.std(cluster_sizes)
    else:
        metrics.update({
            'avg_cluster_size': 0,
            'median_cluster_size': 0,
            'min_cluster_size': 0,
            'max_cluster_size': 0,
            'cluster_size_std': 0
        })
    
    # Silhouette score (if we have valid clusters)
    if n_clusters > 1 and n_outliers < n_total:
        try:
            # Filter out outliers for silhouette calculation
            valid_indices = cluster_labels != -1
            if np.sum(valid_indices) > 1:
                silhouette_avg = silhouette_score(
                    distance_matrix[valid_indices][:, valid_indices], 
                    cluster_labels[valid_indices],
                    metric='precomputed'
                )
                metrics['silhouette_score'] = silhouette_avg
            else:
                metrics['silhouette_score'] = None
        except Exception as e:
            logging.warning(f"Could not calculate silhouette score: {e}")
            metrics['silhouette_score'] = None
    else:
        metrics['silhouette_score'] = None
    
    return metrics

def calculate_terminal_metrics(terminals_df: pd.DataFrame, clustered_endpoints: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate metrics for terminal discovery quality.

    Args:
        terminals_df (pd.DataFrame): Terminal summary data.
        clustered_endpoints (pd.DataFrame): Raw endpoint data with cluster assignments.

    Returns:
        Dict[str, Any]: Dictionary of terminal metrics.
    """
    metrics = {}
    
    if terminals_df.empty:
        return {'n_terminals': 0, 'error': 'No terminals found'}
    
    # Basic terminal statistics
    metrics['n_terminals'] = len(terminals_df)
    metrics['total_visits'] = terminals_df['total_visits'].sum()
    metrics['avg_visits_per_terminal'] = terminals_df['total_visits'].mean()
    metrics['unique_vessels_total'] = clustered_endpoints['mmsi'].nunique()
    
    # Terminal activity distribution
    metrics['most_active_terminal_visits'] = terminals_df['total_visits'].max()
    metrics['least_active_terminal_visits'] = terminals_df['total_visits'].min()
    metrics['terminal_activity_std'] = terminals_df['total_visits'].std()
    
    # Geographic spread
    metrics['lat_range'] = terminals_df['centroid_lat'].max() - terminals_df['centroid_lat'].min()
    metrics['lon_range'] = terminals_df['centroid_lon'].max() - terminals_df['centroid_lon'].min()
    metrics['avg_terminal_precision_lat'] = terminals_df['lat_std'].mean()
    metrics['avg_terminal_precision_lon'] = terminals_df['lon_std'].mean()
    
    return metrics

def calculate_route_metrics(routes_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate metrics for route discovery quality.

    Args:
        routes_df (pd.DataFrame): Route data with metadata.

    Returns:
        Dict[str, Any]: Dictionary of route metrics.
    """
    metrics = {}
    
    if routes_df.empty:
        return {'n_routes': 0, 'error': 'No routes found'}
    
    # Basic route statistics
    metrics['n_routes'] = len(routes_df)
    metrics['total_journeys_in_routes'] = routes_df['total_journeys'].sum()
    metrics['avg_journeys_per_route'] = routes_df['total_journeys'].mean()
    metrics['unique_vessels_in_routes'] = routes_df['unique_vessels'].sum()
    
    # Route usage distribution
    metrics['most_used_route_journeys'] = routes_df['total_journeys'].max()
    metrics['least_used_route_journeys'] = routes_df['total_journeys'].min()
    metrics['route_usage_std'] = routes_df['total_journeys'].std()
    
    # Duration analysis
    if 'avg_duration_hours' in routes_df.columns and routes_df['avg_duration_hours'].notna().any():
        duration_data = routes_df['avg_duration_hours'].dropna()
        metrics['avg_route_duration_hours'] = duration_data.mean()
        metrics['min_route_duration_hours'] = duration_data.min()
        metrics['max_route_duration_hours'] = duration_data.max()
        metrics['route_duration_std'] = duration_data.std()
    
    return metrics

def log_validation_summary(clustering_metrics: Dict[str, Any], 
                         terminal_metrics: Dict[str, Any], 
                         route_metrics: Dict[str, Any]) -> None:
    """
    Logs a comprehensive summary of lane discovery validation metrics.

    Args:
        clustering_metrics (Dict[str, Any]): Clustering quality metrics.
        terminal_metrics (Dict[str, Any]): Terminal discovery metrics.
        route_metrics (Dict[str, Any]): Route discovery metrics.
    """
    logging.info("=" * 60)
    logging.info("LANE DISCOVERY VALIDATION SUMMARY")
    logging.info("=" * 60)
    
    # Clustering results
    logging.info(f"CLUSTERING RESULTS:")
    logging.info(f"  • Found {clustering_metrics.get('n_clusters', 0)} route clusters")
    logging.info(f"  • {clustering_metrics.get('n_outliers', 0)} outlier journeys ({clustering_metrics.get('outlier_percentage', 0):.1f}%)")
    logging.info(f"  • Average cluster size: {clustering_metrics.get('avg_cluster_size', 0):.1f} journeys")
    if clustering_metrics.get('silhouette_score') is not None:
        logging.info(f"  • Silhouette score: {clustering_metrics['silhouette_score']:.3f}")
    
    # Terminal results
    logging.info(f"TERMINAL DISCOVERY:")
    logging.info(f"  • Found {terminal_metrics.get('n_terminals', 0)} terminals")
    logging.info(f"  • Total visits: {terminal_metrics.get('total_visits', 0)}")
    logging.info(f"  • Average visits per terminal: {terminal_metrics.get('avg_visits_per_terminal', 0):.1f}")
    logging.info(f"  • Geographic coverage: {terminal_metrics.get('lat_range', 0):.1f}° lat × {terminal_metrics.get('lon_range', 0):.1f}° lon")
    
    # Route results
    logging.info(f"ROUTE DISCOVERY:")
    logging.info(f"  • Found {route_metrics.get('n_routes', 0)} shipping lanes")
    logging.info(f"  • Total journeys captured: {route_metrics.get('total_journeys_in_routes', 0)}")
    logging.info(f"  • Average journeys per route: {route_metrics.get('avg_journeys_per_route', 0):.1f}")
    if route_metrics.get('avg_route_duration_hours') is not None:
        logging.info(f"  • Average route duration: {route_metrics['avg_route_duration_hours']:.1f} hours")
    
    logging.info("=" * 60)
