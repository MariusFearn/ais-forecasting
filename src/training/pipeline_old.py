"""
Standardized training pipeline for vessel trajectory prediction.

This module centralizes all training logic that was scattered across multiple scripts,
solving the code duplication and inconsistency issues discovered during refactoring.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging
from datetime import datetime
import warnings
import yaml
import os

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Local imports
from src.data.preprocessing import DataPreprocessor, preprocess_vessel_data
from src.utils.metrics import calculate_h3_distance_error
from src.features.vessel_features import VesselFeatureEngine

warnings.filterwarnings('ignore')


class TrainingPipeline:
    """
    Common utilities and pipeline components for training workflows.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path) if config_path else {}
        self.setup_logging()
    
    def setup_logging(self, log_level: str = "INFO"):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            logging.warning(f"Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logging.info(f"Configuration loaded from: {config_path}")
        return config
    
    def validate_data_paths(self, data_dir: str) -> bool:
        """
        Validate that required data directories exist.
        
        Args:
            data_dir: Root data directory
            
        Returns:
            True if all paths exist
        """
        required_dirs = [
            os.path.join(data_dir, 'raw'),
            os.path.join(data_dir, 'processed'),
            os.path.join(data_dir, 'models')
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                logging.error(f"Required directory missing: {dir_path}")
                return False
        
        logging.info("All required data directories exist")
        return True
    
    def create_output_directories(self, base_dir: str, subdirs: list = None):
        """
        Create output directories for training artifacts.
        
        Args:
            base_dir: Base output directory
            subdirs: List of subdirectories to create
        """
        if subdirs is None:
            subdirs = ['models', 'logs', 'results', 'plots']
        
        for subdir in subdirs:
            dir_path = os.path.join(base_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            logging.debug(f"Created directory: {dir_path}")
    
    def get_default_paths(self, data_dir: str = None) -> Dict[str, str]:
        """
        Get default paths for common training files.
        
        Args:
            data_dir: Root data directory
            
        Returns:
            Dictionary of default paths
        """
        if data_dir is None:
            data_dir = '/home/marius/repo_linux/ais-forecasting/data'
        
        return {
            'simple_training_data': f'{data_dir}/processed/training_sets/simple_h3_sequences.pkl',
            'multi_vessel_training_data': f'{data_dir}/processed/training_sets/multi_vessel_h3_sequences.pkl',
            'simple_model_dir': f'{data_dir}/models/final_models',
            'enhanced_model_dir': f'{data_dir}/models/final_models',
            'raw_data_dir': f'{data_dir}/raw',
            'processed_data_dir': f'{data_dir}/processed'
        }
    
    def log_training_start(self, model_type: str, config: Dict[str, Any] = None):
        """
        Log training start information.
        
        Args:
            model_type: Type of model being trained
            config: Training configuration
        """
        logging.info("=" * 60)
        logging.info(f"🚀 STARTING {model_type.upper()} MODEL TRAINING")
        logging.info("=" * 60)
        
        if config:
            logging.info("Training Configuration:")
            for key, value in config.items():
                logging.info(f"   {key}: {value}")
    
    def log_training_complete(self, model_type: str, metrics: Dict[str, Any] = None, 
                            save_paths: Dict[str, str] = None):
        """
        Log training completion information.
        
        Args:
            model_type: Type of model that was trained
            metrics: Training metrics
            save_paths: Paths where model was saved
        """
        logging.info("=" * 60)
        logging.info(f"✅ {model_type.upper()} MODEL TRAINING COMPLETE")
        logging.info("=" * 60)
        
        if metrics:
            logging.info("Final Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logging.info(f"   {key}: {value:.4f}")
                else:
                    logging.info(f"   {key}: {value}")
        
        if save_paths:
            logging.info("Saved Files:")
            for name, path in save_paths.items():
                logging.info(f"   {name}: {path}")
    
    def validate_training_data(self, training_path: str) -> bool:
        """
        Validate training data file exists and is readable.
        
        Args:
            training_path: Path to training data file
            
        Returns:
            True if valid
        """
        if not os.path.exists(training_path):
            logging.error(f"Training data file not found: {training_path}")
            return False
        
        try:
            import pandas as pd
            df = pd.read_pickle(training_path)
            logging.info(f"Training data validated: {len(df)} samples")
            return True
        except Exception as e:
            logging.error(f"Failed to load training data: {e}")
            return False
    
    def get_training_summary(self, training_df) -> Dict[str, Any]:
        """
        Generate summary statistics for training data.
        
        Args:
            training_df: Training DataFrame
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'total_samples': len(training_df),
            'features': list(training_df.columns),
            'memory_usage_mb': training_df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Add vessel-specific stats if vessel column exists
        if 'vessel_imo' in training_df.columns:
            summary['unique_vessels'] = training_df['vessel_imo'].nunique()
            summary['avg_samples_per_vessel'] = len(training_df) / summary['unique_vessels']
        
        # Add H3 cell stats if present
        if 'current_h3_cell' in training_df.columns:
            summary['unique_h3_cells'] = training_df['current_h3_cell'].nunique()
        
        if 'target_h3_cell' in training_df.columns:
            summary['unique_target_cells'] = training_df['target_h3_cell'].nunique()
        
        return summary
