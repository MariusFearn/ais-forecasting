"""
Model Evaluation Module

This module contains the core evaluation logic extracted from scripts/evaluate.py
to follow the src/scripts convention. Scripts should be thin wrappers that call these functions.
"""

import os
import logging
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

# Import our existing modules
from ..data.loader import AISDataLoader
from .tft_model import TFTModel
from .nbeats_model import NBeatsModel
from ..utils.metrics import default_metrics
from ..visualization.plots import plot_forecast, plot_error_distribution


class ModelEvaluator:
    """
    Handles model evaluation logic that was previously in scripts/evaluate.py
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model evaluator.
        
        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.data_loader = AISDataLoader(data_dir=config.get('data_dir', './data'))
        
    def load_model(self, model_type: str, model_path: str, training_dataset=None):
        """
        Load a trained model of the specified type.
        
        Args:
            model_type: Type of model ('tft' or 'nbeats')
            model_path: Path to the trained model
            training_dataset: Training dataset needed for model loading
            
        Returns:
            Loaded model instance
        """
        if model_type.lower() == 'tft':
            return TFTModel.load(model_path, self.config['model'], training_dataset)
        elif model_type.lower() == 'nbeats':
            return NBeatsModel.load(model_path, self.config['model'], training_dataset)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_dataset(self, df_data: pd.DataFrame):
        """
        Create training and test datasets from the provided data.
        
        Args:
            df_data: DataFrame with the data
            
        Returns:
            Tuple of (training_dataset, test_dataset)
        """
        return self.data_loader.create_time_series_dataset(
            df_data,
            time_idx=self.config['data']['time_idx_column'],
            target=self.config['data']['target_column'],
            group_ids=self.config['data']['group_id_columns'],
            max_encoder_length=self.config['model']['max_encoder_length'],
            max_prediction_length=self.config['model']['max_prediction_length'],
            time_varying_known_reals=self.config['features']['time_varying_known_reals'],
            time_varying_unknown_reals=self.config['features']['time_varying_unknown_reals'],
        )
    
    def evaluate_model(self, model_type: str, model_path: str, test_data_path: str) -> Dict[str, float]:
        """
        Evaluate a trained model and return metrics.
        
        Args:
            model_type: Type of model ('tft' or 'nbeats')
            model_path: Path to the trained model
            test_data_path: Path to the test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        logging.info("Loading test data...")
        df_test = self.data_loader.load_processed_data(test_data_path)
        
        # Create datasets
        training, test_dataset = self.create_dataset(df_test)
        
        # Load model
        logging.info(f"Loading {model_type} model...")
        model = self.load_model(model_type, model_path, training)
        
        # Create test dataloader
        test_dataloader = test_dataset.to_dataloader(
            train=False, 
            batch_size=self.config['model']['batch_size'],
            num_workers=0
        )
        
        # Evaluate model
        logging.info("Evaluating model...")
        metrics = model.evaluate(test_dataloader)
        
        return metrics
    
    def generate_predictions(self, model_type: str, model_path: str, test_data_path: str):
        """
        Generate predictions for visualization.
        
        Args:
            model_type: Type of model ('tft' or 'nbeats')
            model_path: Path to the trained model
            test_data_path: Path to the test data
            
        Returns:
            Tuple of (predictions, actuals)
        """
        df_test = self.data_loader.load_processed_data(test_data_path)
        training, test_dataset = self.create_dataset(df_test)
        model = self.load_model(model_type, model_path, training)
        
        test_dataloader = test_dataset.to_dataloader(
            train=False, 
            batch_size=self.config['model']['batch_size'],
            num_workers=0
        )
        
        # Generate predictions
        predictions = model.predict(test_dataloader)
        actuals = torch.cat([y for x, (y, weight) in iter(test_dataloader)])
        
        return predictions, actuals
    
    def generate_evaluation_report(self, model_type: str, model_path: str, test_data_path: str, output_dir: str):
        """
        Generate a comprehensive evaluation report with metrics and visualizations.
        
        Args:
            model_type: Type of model ('tft' or 'nbeats')
            model_path: Path to the trained model
            test_data_path: Path to the test data
            output_dir: Directory to save evaluation results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get evaluation metrics
        metrics = self.evaluate_model(model_type, model_path, test_data_path)
        
        # Generate predictions for visualization
        predictions, actuals = self.generate_predictions(model_type, model_path, test_data_path)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(output_dir, f"{model_type}_evaluation_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        # Generate and save plots
        self._save_forecast_plot(actuals, predictions, model_type, output_dir)
        self._save_error_distribution_plot(actuals, predictions, model_type, output_dir)
        
        # Print results
        self._print_evaluation_results(metrics, output_dir)
        
        return metrics
    
    def _save_forecast_plot(self, actuals, predictions, model_type: str, output_dir: str):
        """Save forecast vs actual plot."""
        forecast_fig = plot_forecast(
            actuals.numpy(), 
            predictions.numpy(),
            title=f"{model_type.upper()} Model Forecast vs Actual"
        )
        forecast_path = os.path.join(output_dir, f"{model_type}_forecast_plot.png")
        forecast_fig.savefig(forecast_path, dpi=300, bbox_inches='tight')
        plt.close(forecast_fig)
    
    def _save_error_distribution_plot(self, actuals, predictions, model_type: str, output_dir: str):
        """Save error distribution plot."""
        error_fig = plot_error_distribution(
            actuals.numpy(),
            predictions.numpy(),
            title=f"{model_type.upper()} Model Error Distribution"
        )
        error_path = os.path.join(output_dir, f"{model_type}_error_distribution.png")
        error_fig.savefig(error_path, dpi=300, bbox_inches='tight')
        plt.close(error_fig)
    
    def _print_evaluation_results(self, metrics: Dict[str, float], output_dir: str):
        """Print evaluation results to console."""
        logging.info("Evaluation Results:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value:.4f}")
        
        logging.info(f"Evaluation results saved to: {output_dir}")
