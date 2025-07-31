from .metrics import (
    mae, rmse, smape, mape, quantile_loss, 
    MetricCollection, default_metrics
)

# Optional import for optimization features
try:
    from .optimize import HyperparameterOptimization, create_optuna_callbacks
    OPTIMIZATION_AVAILABLE = True
    optimize_exports = ['HyperparameterOptimization', 'create_optuna_callbacks']
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    optimize_exports = []

__all__ = [
    'mae', 'rmse', 'smape', 'mape', 'quantile_loss',
    'MetricCollection', 'default_metrics',
] + optimize_exports
