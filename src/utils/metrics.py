import torch
import numpy as np
from typing import Union, Callable, Dict, Any


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
