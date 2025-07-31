#!/usr/bin/env python3
"""
Prediction script for AIS forecasting models.

This is now a thin wrapper around src/models/prediction.py
following the src/scripts convention.
"""

import argparse
import sys
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.prediction import ModelPredictor


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
    
    # Create predictor and generate predictions
    try:
        predictor = ModelPredictor(config)
        predictions = predictor.predict(
            args.model_type, 
            args.model_path, 
            args.data, 
            args.output
        )
        logging.info("Prediction completed successfully!")
        return predictions
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
