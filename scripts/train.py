#!/usr/bin/env python3
"""
Training script for AIS forecasting models.
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.loader import AISDataLoader
from src.models.tft_model import TFTModel
from src.models.nbeats_model import NBeatsModel
from src.utils.metrics import default_metrics


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


def train_model(config: dict, model_type: str, data_path: str):
    """
    Train a model with the given configuration.
    
    Args:
        config: Configuration dictionary
        model_type: Type of model to train ('tft' or 'nbeats')
        data_path: Path to the training data
    """
    # Setup data loader
    data_loader = AISDataLoader(data_dir=config['data']['data_dir'])
    
    # Load processed data
    df_data = data_loader.load_processed_data(data_path)
    
    # Initialize model
    if model_type.lower() == 'tft':
        model = TFTModel(config['model'])
    elif model_type.lower() == 'nbeats':
        model = NBeatsModel(config['model'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    logging.info(f"Starting training for {model_type} model...")
    model.fit(df_data=df_data)
    
    # Save the model
    model_dir = config.get('model_output_dir', './data/models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_type}_model.pth")
    model.save(model_path)
    
    logging.info(f"Model saved to: {model_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train AIS forecasting models")
    parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    parser.add_argument("--model", "-m", required=True, choices=['tft', 'nbeats'], 
                      help="Model type to train")
    parser.add_argument("--data", "-d", required=True, help="Path to training data")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Train model
    try:
        train_model(config, args.model, args.data)
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
