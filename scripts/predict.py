#!/usr/bin/env python3
"""
Prediction script for AIS forecasting models.
"""

import argparse
import os
import sys
import yaml
import logging
import pandas as pd
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.loader import AISDataLoader
from src.models.tft_model import TFTModel
from src.models.nbeats_model import NBeatsModel


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def predict(config: dict, model_type: str, model_path: str, data_path: str, output_path: str):
    """
    Generate predictions using a trained model.
    
    Args:
        config: Configuration dictionary
        model_type: Type of model ('tft' or 'nbeats')
        model_path: Path to the trained model
        data_path: Path to the data for prediction
        output_path: Path to save predictions
    """
    # Setup data loader
    data_loader = AISDataLoader(data_dir=config.get('data_dir', './data'))
    
    # Load data
    df_data = data_loader.load_processed_data(data_path)
    
    # Load model
    if model_type.lower() == 'tft':
        # For TFT, we need the training dataset to load the model
        training, _ = data_loader.create_time_series_dataset(
            df_data,
            time_idx=config['data']['time_idx_column'],
            target=config['data']['target_column'],
            group_ids=config['data']['group_id_columns'],
            max_encoder_length=config['model']['max_encoder_length'],
            max_prediction_length=config['model']['max_prediction_length'],
            time_varying_known_reals=config['features']['time_varying_known_reals'],
            time_varying_unknown_reals=config['features']['time_varying_unknown_reals'],
        )
        model = TFTModel.load(model_path, config['model'], training)
    elif model_type.lower() == 'nbeats':
        # For N-BEATS, we need the training dataset to load the model
        training, _ = data_loader.create_time_series_dataset(
            df_data,
            time_idx=config['data']['time_idx_column'],
            target=config['data']['target_column'],
            group_ids=config['data']['group_id_columns'],
            max_encoder_length=config['model']['max_encoder_length'],
            max_prediction_length=config['model']['max_prediction_length'],
            time_varying_known_reals=config['features']['time_varying_known_reals'],
            time_varying_unknown_reals=config['features']['time_varying_unknown_reals'],
        )
        model = NBeatsModel.load(model_path, config['model'], training)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create test dataset
    _, test_dataset = data_loader.create_time_series_dataset(
        df_data,
        time_idx=config['data']['time_idx_column'],
        target=config['data']['target_column'],
        group_ids=config['data']['group_id_columns'],
        max_encoder_length=config['model']['max_encoder_length'],
        max_prediction_length=config['model']['max_prediction_length'],
        time_varying_known_reals=config['features']['time_varying_known_reals'],
        time_varying_unknown_reals=config['features']['time_varying_unknown_reals'],
    )
    
    # Create dataloader
    test_dataloader = test_dataset.to_dataloader(
        train=False, 
        batch_size=config['model']['batch_size'],
        num_workers=0
    )
    
    # Generate predictions
    logging.info("Generating predictions...")
    predictions = model.predict(test_dataloader)
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(predictions, output_path)
    
    logging.info(f"Predictions saved to: {output_path}")
    
    return predictions


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Generate predictions with AIS forecasting models")
    parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    parser.add_argument("--model-type", "-m", required=True, choices=['tft', 'nbeats'], 
                      help="Model type")
    parser.add_argument("--model-path", "-p", required=True, help="Path to trained model")
    parser.add_argument("--data", "-d", required=True, help="Path to prediction data")
    parser.add_argument("--output", "-o", required=True, help="Path to save predictions")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate predictions
    try:
        predict(config, args.model_type, args.model_path, args.data, args.output)
        logging.info("Prediction completed successfully!")
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
