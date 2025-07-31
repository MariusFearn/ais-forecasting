"""
Enhanced Model Trainer Module

This module contains logic for training enhanced models for vessel trajectory prediction.
This will be used in scripts/train_enhanced_model.py
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Any, List


class EnhancedModelTrainer:
    """
    Trains enhanced models using multi-vessel data and more sophisticated features.
    """
    
    def __init__(self, model_config: Dict[str, Any] = None, data_dir: str = None):
        """
        Initialize the enhanced model trainer.
        
        Args:
            model_config: Configuration for model parameters
            data_dir: Directory containing training data
        """
        self.data_dir = data_dir or '/home/marius/repo_linux/ais-forecasting/data'
        self.model_config = model_config or {
            'n_estimators': 100,
            'max_depth': 15,
            'random_state': 42,
            'n_jobs': -1  # Use all available cores
        }
        self.model = None
        self.h3_encoder = None
        self.vessel_encoder = None
        self.feature_importance = None
    
    def load_training_data(self, training_path: str = None) -> pd.DataFrame:
        """
        Load multi-vessel training data from pickle file.
        
        Args:
            training_path: Path to training data file
            
        Returns:
            DataFrame with training data
        """
        if training_path is None:
            training_path = f'{self.data_dir}/processed/training_sets/multi_vessel_h3_sequences.pkl'
        
        logging.info(f"Loading multi-vessel training data from {training_path}")
        training_df = pd.read_pickle(training_path)
        
        logging.info(f"Loaded {len(training_df):,} training examples")
        logging.info(f"From {training_df['vessel_imo'].nunique()} vessels")
        logging.info(f"Features: {list(training_df.columns[:-1])}")
        
        return training_df
    
    def prepare_enhanced_features(self, training_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare enhanced features including vessel identity and more complex features.
        
        Args:
            training_df: DataFrame with training data
            
        Returns:
            Tuple of (X, y) ready for training
        """
        logging.info("Preparing enhanced features and target...")
        
        # Encode H3 cells
        self.h3_encoder = LabelEncoder()
        all_h3_cells = list(set(
            training_df['current_h3_cell'].tolist() + 
            training_df['target_h3_cell'].tolist()
        ))
        self.h3_encoder.fit(all_h3_cells)
        
        # Encode vessel identities
        self.vessel_encoder = LabelEncoder()
        self.vessel_encoder.fit(training_df['vessel_imo'].unique())
        
        # Enhanced features (X)
        X = training_df[['current_speed', 'current_heading', 'lat', 'lon', 'time_in_current_cell']].copy()
        X['current_h3_encoded'] = self.h3_encoder.transform(training_df['current_h3_cell'])
        X['vessel_encoded'] = self.vessel_encoder.transform(training_df['vessel_imo'])
        
        # Add derived features
        X['speed_heading_interaction'] = X['current_speed'] * np.cos(np.radians(X['current_heading']))
        X['lat_lon_interaction'] = X['lat'] * X['lon']
        
        # Target (y)
        y = self.h3_encoder.transform(training_df['target_h3_cell'])
        
        logging.info(f"Enhanced features shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")
        logging.info(f"Unique H3 cells: {len(self.h3_encoder.classes_)}")
        logging.info(f"Unique vessels: {len(self.vessel_encoder.classes_)}")
        
        return X, y
    
    def split_data_by_vessel(self, training_df: pd.DataFrame, test_vessels_ratio: float = 0.2) -> Tuple:
        """
        Split data ensuring vessel separation between train/test.
        
        Args:
            training_df: DataFrame with training data
            test_vessels_ratio: Ratio of vessels to use for testing
            
        Returns:
            Tuple of train_df, test_df
        """
        logging.info("Splitting data by vessel...")
        
        unique_vessels = training_df['vessel_imo'].unique()
        n_test_vessels = max(1, int(len(unique_vessels) * test_vessels_ratio))
        
        # Randomly select test vessels
        np.random.seed(42)
        test_vessels = np.random.choice(unique_vessels, n_test_vessels, replace=False)
        
        train_df = training_df[~training_df['vessel_imo'].isin(test_vessels)]
        test_df = training_df[training_df['vessel_imo'].isin(test_vessels)]
        
        logging.info(f"Train vessels: {len(unique_vessels) - n_test_vessels}")
        logging.info(f"Test vessels: {n_test_vessels}")
        logging.info(f"Train samples: {len(train_df):,}")
        logging.info(f"Test samples: {len(test_df):,}")
        
        return train_df, test_df
    
    def train_enhanced_model(self, X_train: pd.DataFrame, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train enhanced Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        logging.info("Training Enhanced Random Forest...")
        
        self.model = RandomForestClassifier(
            n_estimators=self.model_config['n_estimators'],
            max_depth=self.model_config['max_depth'],
            random_state=self.model_config['random_state'],
            n_jobs=self.model_config.get('n_jobs', -1)
        )
        
        self.model.fit(X_train, y_train)
        logging.info("Enhanced model trained!")
        
        return self.model
    
    def evaluate_enhanced_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                               y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate enhanced model performance.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        logging.info("Evaluating enhanced model...")
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        logging.info(f"Train Accuracy: {train_accuracy:.3f}")
        logging.info(f"Test Accuracy: {test_accuracy:.3f}")
        
        # Feature importance
        feature_names = X_train.columns
        importances = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logging.info("Enhanced Feature Importance:")
        for _, row in self.feature_importance.head(10).iterrows():
            logging.info(f"   {row['feature']}: {row['importance']:.3f}")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_classes': len(self.h3_encoder.classes_),
            'n_vessels': len(self.vessel_encoder.classes_),
            'n_features': len(feature_names)
        }
    
    def save_enhanced_model(self, model_dir: str = None):
        """
        Save enhanced model and encoders.
        
        Args:
            model_dir: Directory to save model files
        """
        if model_dir is None:
            model_dir = f'{self.data_dir}/models/final_models'
        
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'enhanced_h3_predictor.pkl')
        h3_encoder_path = os.path.join(model_dir, 'enhanced_h3_label_encoder.pkl')
        vessel_encoder_path = os.path.join(model_dir, 'vessel_label_encoder.pkl')
        importance_path = os.path.join(model_dir, 'enhanced_feature_importance.csv')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.h3_encoder, h3_encoder_path)
        joblib.dump(self.vessel_encoder, vessel_encoder_path)
        
        if self.feature_importance is not None:
            self.feature_importance.to_csv(importance_path, index=False)
        
        logging.info(f"Enhanced model saved to: {model_path}")
        logging.info(f"H3 encoder saved to: {h3_encoder_path}")
        logging.info(f"Vessel encoder saved to: {vessel_encoder_path}")
        logging.info(f"Feature importance saved to: {importance_path}")
        
        return model_path, h3_encoder_path, vessel_encoder_path
    
    def train_enhanced_h3_predictor(self, training_path: str = None, 
                                   output_dir: str = None) -> Tuple[RandomForestClassifier, Dict, Dict]:
        """
        Complete training pipeline for enhanced H3 predictor.
        
        Args:
            training_path: Path to training data
            output_dir: Directory to save model
            
        Returns:
            Tuple of (model, encoders_dict, metrics)
        """
        logging.info("🚀 Training Enhanced Multi-Vessel H3 Cell Predictor...")
        
        # Load and prepare data
        training_df = self.load_training_data(training_path)
        
        # Split by vessel to avoid data leakage
        train_df, test_df = self.split_data_by_vessel(training_df)
        
        # Prepare features
        X_train, y_train = self.prepare_enhanced_features(train_df)
        X_test, y_test = self.prepare_enhanced_features(test_df)
        
        # Train model
        self.train_enhanced_model(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_enhanced_model(X_train, X_test, y_train, y_test)
        
        # Save model
        self.save_enhanced_model(output_dir)
        
        # Create encoders dictionary
        encoders = {
            'h3_encoder': self.h3_encoder,
            'vessel_encoder': self.vessel_encoder
        }
        
        logging.info(f"🎉 SUCCESS! Enhanced H3 Predictor trained!")
        logging.info(f"   📊 Test Accuracy: {metrics['test_accuracy']:.1%}")
        logging.info(f"   🚢 Vessels: {metrics['n_vessels']}")
        logging.info(f"   🗺️ H3 Cells: {metrics['n_classes']}")
        logging.info(f"   🔧 Features: {metrics['n_features']}")
        logging.info("🚀 Enhanced model ready for vessel movement prediction!")
        
        return self.model, encoders, metrics
