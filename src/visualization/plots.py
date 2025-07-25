import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple


def setup_plot_style():
    """Set up the default plotting style."""
    sns.set(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })


def plot_forecast(y_true: np.ndarray, 
                 y_pred: np.ndarray, 
                 timestamps: List[Any] = None,
                 prediction_intervals: Optional[Dict[str, np.ndarray]] = None,
                 title: str = "Forecast vs Actual",
                 xlabel: str = "Time",
                 ylabel: str = "Value",
                 figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot forecasted values against actual values.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        timestamps: Timestamp labels for x-axis
        prediction_intervals: Dictionary of prediction intervals (e.g., {"90%": np.array([lower, upper])})
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = timestamps if timestamps is not None else np.arange(len(y_true))
    
    # Plot actual values
    ax.plot(x, y_true, label="Actual", marker="o", linestyle="-", color="blue")
    
    # Plot predicted values
    ax.plot(x, y_pred, label="Forecast", marker="x", linestyle="--", color="red")
    
    # Plot prediction intervals if provided
    if prediction_intervals is not None:
        for interval_name, (lower, upper) in prediction_intervals.items():
            ax.fill_between(x, lower, upper, alpha=0.2, label=f"{interval_name} Interval")
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_error_distribution(y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           title: str = "Error Distribution",
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot the distribution of prediction errors.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    setup_plot_style()
    
    errors = y_pred - y_true
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(errors, kde=True, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    ax.axvline(x=0, color='r', linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_metrics_over_time(metrics_df: pd.DataFrame, 
                          metric_names: List[str],
                          title: str = "Metrics Over Time",
                          xlabel: str = "Time",
                          ylabel: str = "Metric Value",
                          figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot multiple metrics over time.
    
    Args:
        metrics_df: DataFrame containing metrics with a datetime index
        metric_names: List of metric column names to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for metric in metric_names:
        ax.plot(metrics_df.index, metrics_df[metric], label=metric)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names: List[str], 
                           importance_scores: np.ndarray,
                           title: str = "Feature Importance",
                           figsize: Tuple[int, int] = (12, 8),
                           horizontal: bool = True) -> plt.Figure:
    """
    Plot feature importance scores.
    
    Args:
        feature_names: Names of features
        importance_scores: Importance scores for each feature
        title: Plot title
        figsize: Figure size
        horizontal: Whether to plot bars horizontally
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    setup_plot_style()
    
    # Sort features by importance
    sorted_idx = importance_scores.argsort()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if horizontal:
        ax.barh(np.array(feature_names)[sorted_idx], importance_scores[sorted_idx])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
    else:
        ax.bar(np.array(feature_names)[sorted_idx], importance_scores[sorted_idx])
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        plt.xticks(rotation=45, ha="right")
    
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_attention_weights(attention_weights: np.ndarray,
                          encoder_features: List[str],
                          decoder_times: List[str],
                          title: str = "Attention Weights",
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot attention weights from a Temporal Fusion Transformer.
    
    Args:
        attention_weights: 2D array of attention weights
        encoder_features: Names of encoder features
        decoder_times: Labels for decoder time steps
        title: Plot title
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(attention_weights, cmap="viridis")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")
    
    # Set labels
    ax.set_xticks(np.arange(len(decoder_times)))
    ax.set_yticks(np.arange(len(encoder_features)))
    ax.set_xticklabels(decoder_times)
    ax.set_yticklabels(encoder_features)
    
    # Rotate the x-axis labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_title(title)
    fig.tight_layout()
    
    return fig
