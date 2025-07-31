#!/usr/bin/env python3
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
    Comprehensive training pipeline for vessel trajectory prediction.
    
    Centralizes all the training logic that was duplicated across scripts:
    - Data loading and preprocessing
    - Feature engineering
    - Model training with multiple algorithms
    - Evaluation and metrics calculation
    - Model saving and metadata management
    """
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 model_save_dir: str = "data/models/final_models",
                 memory_optimize: bool = True,
                 verbose: bool = True,
                 config_path: Optional[str] = None):
        """
        Initialize the training pipeline.
        
        Args:
            data_path: Path to training data file
            model_save_dir: Directory to save trained models
            memory_optimize: Whether to optimize memory usage
            verbose: Whether to print detailed progress
            config_path: Path to configuration file
        """
        self.data_path = data_path
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.memory_optimize = memory_optimize
        self.verbose = verbose
        
        # Load config if provided
        self.config = self.load_config(config_path) if config_path else {}
        
        # Initialize components
        self.preprocessor = DataPreprocessor(memory_optimize=memory_optimize, verbose=verbose)
        self.feature_engine = VesselFeatureEngine()
        
        # Training state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.target_encoder = None
        self.feature_metadata = None
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
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
            return yaml.safe_load(f)
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load training data from pickle file.
        
        Args:
            data_path: Path to data file (overrides instance path)
            
        Returns:
            Loaded DataFrame
        """
        path = data_path or self.data_path
        if not path:
            raise ValueError("No data path provided")
            
        if self.verbose:
            print(f"📂 Loading data from: {path}")
            
        data = pd.read_pickle(path)
        
        if self.verbose:
            print(f"   ✅ Loaded {len(data):,} records")
            print(f"   📊 Shape: {data.shape}")
            print(f"   💾 Memory: {data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
        self.data = data
        return data
        
    def prepare_features_and_target(self, 
                                   target_column: str = 'target_h3_cell',
                                   exclude_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from loaded data.
        
        Args:
            target_column: Name of target column
            exclude_columns: Additional columns to exclude from features
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if self.verbose:
            print(f"🎯 Preparing features and target...")
            print(f"   Target column: {target_column}")
            
        # Check if target exists
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        # Prepare exclusion list
        default_excludes = [target_column, 'vessel_imo', 'timestamp']
        exclude_columns = exclude_columns or []
        all_excludes = list(set(default_excludes + exclude_columns))
        
        if self.verbose:
            print(f"   Excluding columns: {all_excludes}")
            
        # Extract target
        y = self.data[target_column].copy()
        
        # Process features using comprehensive preprocessor
        X, self.feature_metadata = self.preprocessor.process_features(
            self.data, 
            exclude_columns=all_excludes
        )
        
        if self.verbose:
            print(f"   ✅ Features shape: {X.shape}")
            print(f"   🎯 Target unique values: {y.nunique():,}")
            print(f"   📊 Target sample: {y.value_counts().head()}")
            
        return X, y
        
    def encode_target(self, y: pd.Series) -> pd.Series:
        """
        Encode target column for classification.
        
        Args:
            y: Target series
            
        Returns:
            Encoded target series
        """
        if self.verbose:
            print(f"🏷️  Encoding target column...")
            
        # Handle string targets (H3 cells)
        if y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y.astype(str))
            
            if self.verbose:
                print(f"   ✅ Encoded {len(self.target_encoder.classes_)} unique target classes")
                
            return pd.Series(y_encoded, index=y.index)
        else:
            if self.verbose:
                print(f"   ✅ Target already numeric")
            return y
            
    def split_data(self, 
                   X: pd.DataFrame, 
                   y: pd.Series,
                   test_size: float = 0.2,
                   random_state: int = 42) -> None:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        if self.verbose:
            print(f"✂️  Splitting data ({test_size:.0%} test, {1-test_size:.0%} train)...")
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        if self.verbose:
            print(f"   ✅ Train: {len(self.X_train):,} samples")
            print(f"   ✅ Test:  {len(self.X_test):,} samples")
            
    def train_xgboost_model(self, 
                           model_params: Optional[Dict] = None,
                           feature_selection: bool = True,
                           max_features: int = 25) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier with feature selection.
        
        Args:
            model_params: XGBoost parameters
            feature_selection: Whether to perform feature selection
            max_features: Maximum number of features to select
            
        Returns:
            Trained XGBoost model
        """
        if self.X_train is None:
            raise ValueError("Data not split. Call split_data() first.")
            
        if self.verbose:
            print(f"🚀 Training XGBoost model...")
            
        # Default parameters optimized for large datasets
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        }
        
        if model_params:
            default_params.update(model_params)
            
        # Feature selection
        if feature_selection and len(self.X_train.columns) > max_features:
            if self.verbose:
                print(f"   🔍 Performing feature selection ({len(self.X_train.columns)} → {max_features})...")
                
            # Train a simple model for feature importance
            temp_model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
            temp_model.fit(self.X_train, self.y_train)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': temp_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Select top features
            selected_features = feature_importance.head(max_features)['feature'].tolist()
            
            # Update training data
            self.X_train = self.X_train[selected_features]
            self.X_test = self.X_test[selected_features]
            
            if self.verbose:
                print(f"   ✅ Selected {len(selected_features)} most important features")
                print(f"   🏆 Top 5: {selected_features[:5]}")
        
        # Train XGBoost
        if self.verbose:
            print(f"   🎯 Training with {len(self.X_train.columns)} features...")
            
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(
            self.X_train, 
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        if self.verbose:
            print(f"   ✅ Model training complete!")
            
        return self.model
        
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model trained. Call train_xgboost_model() first.")
            
        if self.verbose:
            print(f"📊 Evaluating model performance...")
            
        # Predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Basic accuracy
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        
        # H3 distance error (if target encoder exists)
        if self.target_encoder is not None:
            # Decode predictions back to H3 cells
            y_true_h3 = self.target_encoder.inverse_transform(self.y_test)
            y_pred_h3 = self.target_encoder.inverse_transform(y_pred_test)
            
            # Calculate H3 distance error
            avg_distance_error = calculate_h3_distance_error(y_true_h3, y_pred_h3)
        else:
            avg_distance_error = None
            
        # Prepare evaluation results
        evaluation = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'accuracy_difference': train_accuracy - test_accuracy,
            'avg_distance_error_km': avg_distance_error,
            'model_type': 'XGBoost',
            'n_features': len(self.X_train.columns),
            'n_classes': len(np.unique(self.y_train)),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        
        if self.verbose:
            print(f"   🎯 Training Accuracy: {train_accuracy:.1%}")
            print(f"   🎯 Test Accuracy: {test_accuracy:.1%}")
            if avg_distance_error:
                print(f"   📍 Average Distance Error: {avg_distance_error:.2f} km")
            print(f"   🔢 Features: {evaluation['n_features']}")
            print(f"   🏷️  Classes: {evaluation['n_classes']:,}")
            
        return evaluation
        
    def save_model(self, 
                   model_name: str,
                   evaluation_results: Optional[Dict] = None) -> str:
        """
        Save trained model and metadata.
        
        Args:
            model_name: Name for the saved model
            evaluation_results: Evaluation metrics to save
            
        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
            
        if self.verbose:
            print(f"💾 Saving model as: {model_name}")
            
        # Create timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main model
        model_path = self.model_save_dir / f"{model_name}.pkl"
        joblib.dump(self.model, model_path)
        
        # Save target encoder if exists
        if self.target_encoder is not None:
            encoder_path = self.model_save_dir / f"{model_name}_target_encoder.pkl"
            joblib.dump(self.target_encoder, encoder_path)
            
        # Save feature preprocessor
        preprocessor_path = self.model_save_dir / f"{model_name}_preprocessor.pkl"
        self.preprocessor.save_encoders(str(preprocessor_path))
        
        # Save comprehensive metadata
        metadata = {
            'model_name': model_name,
            'training_timestamp': timestamp,
            'model_type': 'XGBoost',
            'feature_columns': list(self.X_train.columns),
            'n_features': len(self.X_train.columns),
            'n_classes': len(np.unique(self.y_train)) if self.target_encoder else None,
            'target_encoder_classes': self.target_encoder.classes_.tolist() if self.target_encoder else None,
            'preprocessing_metadata': self.feature_metadata,
            'evaluation_results': evaluation_results,
            'data_shapes': {
                'train': self.X_train.shape,
                'test': self.X_test.shape
            }
        }
        
        metadata_path = self.model_save_dir / f"{model_name}_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        if self.verbose:
            print(f"   ✅ Model: {model_path}")
            print(f"   ✅ Encoder: {encoder_path if self.target_encoder else 'N/A'}")
            print(f"   ✅ Preprocessor: {preprocessor_path}")
            print(f"   ✅ Metadata: {metadata_path}")
            
        return str(model_path)
        
    def full_training_pipeline(self,
                              data_path: Optional[str] = None,
                              model_name: str = "h3_predictor",
                              target_column: str = 'target_h3_cell',
                              test_size: float = 0.2,
                              feature_selection: bool = True,
                              max_features: int = 25) -> Dict[str, Any]:
        """
        Complete training pipeline from data loading to model saving.
        
        Args:
            data_path: Path to training data
            model_name: Name for saved model
            target_column: Target column name
            test_size: Test set proportion
            feature_selection: Whether to select features
            max_features: Maximum features to select
            
        Returns:
            Complete training results and metadata
        """
        if self.verbose:
            print("🚀 Starting Full Training Pipeline")
            print("=" * 50)
            
        # Step 1: Load data
        self.load_data(data_path)
        
        # Step 2: Prepare features and target
        X, y = self.prepare_features_and_target(target_column)
        
        # Step 3: Encode target
        y_encoded = self.encode_target(y)
        
        # Step 4: Split data
        self.split_data(X, y_encoded, test_size=test_size)
        
        # Step 5: Train model
        self.train_xgboost_model(feature_selection=feature_selection, max_features=max_features)
        
        # Step 6: Evaluate model
        evaluation = self.evaluate_model()
        
        # Step 7: Save model
        model_path = self.save_model(model_name, evaluation)
        
        # Prepare final results
        results = {
            'model_path': model_path,
            'evaluation': evaluation,
            'pipeline_summary': {
                'data_path': data_path or self.data_path,
                'model_name': model_name,
                'target_column': target_column,
                'preprocessing_steps': self.feature_metadata['processing_steps'] if self.feature_metadata else [],
                'training_completed': datetime.now().isoformat()
            }
        }
        
        if self.verbose:
            print("🎉 Training Pipeline Complete!")
            print(f"   Model saved to: {model_path}")
            print(f"   Test Accuracy: {evaluation['test_accuracy']:.1%}")
            if evaluation.get('avg_distance_error_km'):
                print(f"   Distance Error: {evaluation['avg_distance_error_km']:.2f} km")
                
        return results


# Convenience function for quick training
def train_vessel_predictor(data_path: str,
                          model_name: str = "vessel_h3_predictor",
                          memory_optimize: bool = True,
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Quick training function for vessel trajectory prediction.
    
    Args:
        data_path: Path to training data pickle file
        model_name: Name for the saved model
        memory_optimize: Whether to optimize memory usage
        verbose: Whether to print progress
        
    Returns:
        Training results and model metadata
    """
    pipeline = TrainingPipeline(
        data_path=data_path,
        memory_optimize=memory_optimize,
        verbose=verbose
    )
    
    return pipeline.full_training_pipeline(model_name=model_name)


if __name__ == "__main__":
    # Example usage
    print("🚀 Vessel Trajectory Training Pipeline")
    print("=" * 50)
    
    print("✅ Key Features:")
    print("   - Standardized data preprocessing")
    print("   - Automated feature selection")
    print("   - XGBoost training with optimization")
    print("   - Comprehensive evaluation metrics")
    print("   - Complete model persistence")
    
    print("\n🎯 Solves Script Issues:")
    print("   - Eliminates code duplication")
    print("   - Centralized training logic")
    print("   - Consistent preprocessing")
    print("   - Standardized model saving")
    print("   - Unified evaluation metrics")
    
    print("\n📝 Usage:")
    print("   from src.training.pipeline import train_vessel_predictor")
    print("   results = train_vessel_predictor('data/processed/training_data.pkl')")
