from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Tuple


class BaseTimeSeriesModel(ABC):
    """Abstract base class for all time series forecasting models."""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model with the given configuration."""
        pass
    
    @abstractmethod
    def fit(self, train_dataloader, val_dataloader):
        """Train the model using the provided training and validation data."""
        pass
    
    @abstractmethod
    def predict(self, dataloader) -> torch.Tensor:
        """Generate forecasts for the provided data."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the model to the specified path."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """Load the model from the specified path."""
        pass
    
    @abstractmethod
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate the model on the provided data and return metrics."""
        pass
