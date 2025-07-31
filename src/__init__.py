"""
AIS Forecasting Package

A comprehensive deep learning forecasting system for maritime AIS data.
"""

__version__ = "0.1.0"
__author__ = "AIS Forecasting Team"

# Package-level imports for convenience
from src.data import loader, preprocessing
from src.features import geo_features, time_features
from src.models import base_model, tft_model, nbeats_model
from src.utils import metrics
try:
    from src.utils import optimize
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from src.visualization import plots

__all__ = [
    "loader",
    "preprocessing", 
    "geo_features",
    "time_features",
    "base_model",
    "tft_model", 
    "nbeats_model",
    "metrics",
    "optimize",
    "plots"
]
