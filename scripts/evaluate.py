#!/usr/bin/env python3
"""
Evaluation script for AIS forecasting models.
"""

import argparse
import os
import sys
import yaml
import logging
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.loader import AISDataLoader
from src.models.tft_model import TFTModel
from src.models.nbeats_model import NBeatsModel
from src.utils.metrics import default_metrics
from src.visualization.plots import plot_forecast, plot_error_distribution


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


def evaluate_model(config: dict, model_type: str, model_path: str, test_data_path: str, output_dir: str):
    """
    Evaluate a trained model and generate reports.
    
    Args:
        config: Configuration dictionary
        model_type: Type of model ('tft' or 'nbeats')
        model_path: Path to the trained model
        test_data_path: Path to the test data
        output_dir: Directory to save evaluation results
    """
    # Setup data loader
    data_loader = AISDataLoader(data_dir=config.get('data_dir', './data'))
    
    # Load test data
    df_test = data_loader.load_processed_data(test_data_path)
    
    # Load model
    if model_type.lower() == 'tft':
        # Create training dataset for model loading
        training, _ = data_loader.create_time_series_dataset(
            df_test,
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
        # Create training dataset for model loading
        training, _ = data_loader.create_time_series_dataset(
            df_test,
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
        df_test,
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
    
    # Evaluate model
    logging.info("Evaluating model...")
    metrics = model.evaluate(test_dataloader)
    
    # Generate predictions for visualization
    predictions = model.predict(test_dataloader)
    
    # Extract actual values
    actuals = torch.cat([y for x, (y, weight) in iter(test_dataloader)])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, f"{model_type}_evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Generate and save plots
    # Forecast plot
    forecast_fig = plot_forecast(
        actuals.numpy(), 
        predictions.numpy(),
        title=f"{model_type.upper()} Model Forecast vs Actual"
    )
    forecast_path = os.path.join(output_dir, f"{model_type}_forecast_plot.png")
    forecast_fig.savefig(forecast_path, dpi=300, bbox_inches='tight')
    plt.close(forecast_fig)
    
    # Error distribution plot
    error_fig = plot_error_distribution(
        actuals.numpy(),
        predictions.numpy(),
        title=f"{model_type.upper()} Model Error Distribution"
    )
    error_path = os.path.join(output_dir, f"{model_type}_error_distribution.png")
    error_fig.savefig(error_path, dpi=300, bbox_inches='tight')
    plt.close(error_fig)
    
    # Print metrics
    logging.info("Evaluation Results:")
    for metric_name, metric_value in metrics.items():
        logging.info(f"{metric_name}: {metric_value:.4f}")
    
    logging.info(f"Evaluation results saved to: {output_dir}")
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate AIS forecasting models")
    parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    parser.add_argument("--model-type", "-m", required=True, choices=['tft', 'nbeats'], 
                      help="Model type")
    parser.add_argument("--model-path", "-p", required=True, help="Path to trained model")
    parser.add_argument("--test-data", "-d", required=True, help="Path to test data")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to save evaluation results")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Evaluate model
    try:
        evaluate_model(config, args.model_type, args.model_path, args.test_data, args.output_dir)
        logging.info("Evaluation completed successfully!")
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
