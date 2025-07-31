"""
Model Prediction Module

This module contains the core prediction logic extracted from scripts/predict.py
to follow the src/scripts convention. Scripts should be thin wrappers that call these functions.
"""

import os
import logging
import torch
from pathlib import Path
from typing import Dict, Any

# Import our existing modules
from ..data.loader import AISDataLoader
from .tft_model import TFTModel
from .nbeats_model import NBeatsModel


class ModelPredictor:
    """
    Handles model prediction logic that was previously in scripts/predict.py
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model predictor.
        
        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.data_loader = AISDataLoader(data_dir=config.get('data_dir', './data'))
        self.model = None
        self.model_type = None
    
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
        self.model_type = model_type
        
        if model_type.lower() == 'tft':
            self.model = TFTModel.load(model_path, self.config['model'], training_dataset)
        elif model_type.lower() == 'nbeats':
            self.model = NBeatsModel.load(model_path, self.config['model'], training_dataset)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logging.info(f"{model_type.upper()} model loaded successfully from {model_path}")
        return self.model
    
    def create_dataset_for_prediction(self, df_data):
        """
        Create dataset for prediction from the provided data.
        
        Args:
            df_data: DataFrame with the data for prediction
            
        Returns:
            Tuple of (training_dataset, prediction_dataset)
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
    
    def predict(self, model_type: str, model_path: str, data_path: str, output_path: str):
        """
        Generate predictions using a trained model and save them.
        
        Args:
            model_type: Type of model ('tft' or 'nbeats')
            model_path: Path to the trained model
            data_path: Path to the data for prediction
            output_path: Path to save predictions
            
        Returns:
            Generated predictions tensor
        """
        logging.info("Loading prediction data...")
        df_data = self.data_loader.load_processed_data(data_path)
        
        # Create dataset for prediction
        training, prediction_dataset = self.create_dataset_for_prediction(df_data)
        
        # Load model if not already loaded
        if self.model is None or self.model_type != model_type:
            self.load_model(model_type, model_path, training)
        
        # Create dataloader
        prediction_dataloader = prediction_dataset.to_dataloader(
            train=False, 
            batch_size=self.config['model']['batch_size'],
            num_workers=0
        )
        
        # Generate predictions
        logging.info("Generating predictions...")
        predictions = self.model.predict(prediction_dataloader)
        
        # Save predictions
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(predictions, output_path)
        
        logging.info(f"Predictions saved to: {output_path}")
        
        return predictions
    
    def predict_batch(self, model_type: str, model_path: str, data_paths: list, output_dir: str):
        """
        Generate predictions for multiple data files.
        
        Args:
            model_type: Type of model ('tft' or 'nbeats')
            model_path: Path to the trained model
            data_paths: List of paths to data files for prediction
            output_dir: Directory to save prediction files
            
        Returns:
            Dictionary mapping data_path to prediction_path
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for data_path in data_paths:
            # Generate output filename
            data_filename = Path(data_path).stem
            output_path = os.path.join(output_dir, f"{data_filename}_predictions.pt")
            
            # Generate predictions
            predictions = self.predict(model_type, model_path, data_path, output_path)
            results[data_path] = output_path
            
            logging.info(f"Completed predictions for {data_path}")
        
        logging.info(f"Batch prediction completed. Results saved to {output_dir}")
        return results
    
    def predict_and_evaluate(self, model_type: str, model_path: str, data_path: str, output_path: str):
        """
        Generate predictions and return both predictions and basic evaluation metrics.
        
        Args:
            model_type: Type of model ('tft' or 'nbeats')
            model_path: Path to the trained model
            data_path: Path to the data for prediction
            output_path: Path to save predictions
            
        Returns:
            Tuple of (predictions, evaluation_metrics)
        """
        # Generate predictions
        predictions = self.predict(model_type, model_path, data_path, output_path)
        
        # Load data for evaluation
        df_data = self.data_loader.load_processed_data(data_path)
        training, test_dataset = self.create_dataset_for_prediction(df_data)
        
        # Create dataloader for evaluation
        test_dataloader = test_dataset.to_dataloader(
            train=False, 
            batch_size=self.config['model']['batch_size'],
            num_workers=0
        )
        
        # Basic evaluation
        if hasattr(self.model, 'evaluate'):
            evaluation_metrics = self.model.evaluate(test_dataloader)
        else:
            # Fallback to basic metrics if evaluate method not available
            actuals = torch.cat([y for x, (y, weight) in iter(test_dataloader)])
            mse = torch.mean((predictions - actuals) ** 2).item()
            mae = torch.mean(torch.abs(predictions - actuals)).item()
            evaluation_metrics = {'mse': mse, 'mae': mae}
        
        logging.info("Prediction and evaluation completed")
        for metric_name, metric_value in evaluation_metrics.items():
            logging.info(f"{metric_name}: {metric_value:.4f}")
        
        return predictions, evaluation_metrics
