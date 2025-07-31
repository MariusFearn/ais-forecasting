"""
Simple Model Trainer Module

This module contains logic for training simple models for vessel trajectory prediction.
Extracted from the logic that was previously in scripts/train_simple_model.py
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
from typing import Dict, Tuple, Any


class SimpleModelTrainer:
    """
    Trains simple Random Forest models for H3 cell prediction.
    This logic was previously in scripts/train_simple_model.py
    """
    
    def __init__(self, model_config: Dict[str, Any] = None, data_dir: str = None):
        """
        Initialize the simple model trainer.
        
        Args:
            model_config: Configuration for model parameters
            data_dir: Directory containing training data
        """
        self.data_dir = data_dir or '/home/marius/repo_linux/ais-forecasting/data'
        self.model_config = model_config or {
            'n_estimators': 50,
            'max_depth': 10,
            'random_state': 42
        }
        self.model = None
        self.h3_encoder = None
        self.feature_importance = None
    
    def load_training_data(self, training_path: str = None) -> pd.DataFrame:
        """
        Load training data from pickle file.
        
        Args:
            training_path: Path to training data file
            
        Returns:
            DataFrame with training data
        """
        if training_path is None:
            training_path = f'{self.data_dir}/processed/training_sets/simple_h3_sequences.pkl'
        
        logging.info(f"Loading training data from {training_path}")
        training_df = pd.read_pickle(training_path)
        
        logging.info(f"Loaded {len(training_df)} training examples")
        logging.info(f"Features: {list(training_df.columns[:-1])}")
        
        return training_df
    
    def prepare_features_and_target(self, training_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features (X) and target (y) for model training.
        
        Args:
            training_df: DataFrame with training data
            
        Returns:
            Tuple of (X, y) ready for training
        """
        logging.info("Preparing features and target...")
        
        # Encode H3 cells as numbers (ML models need numbers)
        self.h3_encoder = LabelEncoder()
        all_h3_cells = list(set(
            training_df['current_h3_cell'].tolist() + 
            training_df['target_h3_cell'].tolist()
        ))
        self.h3_encoder.fit(all_h3_cells)
        
        # Features (X)
        X = training_df[['current_speed', 'current_heading', 'lat', 'lon', 'time_in_current_cell']].copy()
        X['current_h3_encoded'] = self.h3_encoder.transform(training_df['current_h3_cell'])
        
        # Target (y)
        y = self.h3_encoder.transform(training_df['target_h3_cell'])
        
        logging.info(f"Features shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")
        logging.info(f"Unique H3 cells: {len(self.h3_encoder.classes_)}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """
        Split data into train/test sets.
        
        Args:
            X: Feature matrix
            y: Target array
            test_size: Fraction for test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logging.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logging.info(f"Train: {len(X_train)} samples")
        logging.info(f"Test: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        logging.info("Training Random Forest...")
        
        self.model = RandomForestClassifier(
            n_estimators=self.model_config['n_estimators'],
            max_depth=self.model_config['max_depth'],
            random_state=self.model_config['random_state']
        )
        
        self.model.fit(X_train, y_train)
        logging.info("Model trained!")
        
        return self.model
    
    def evaluate_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        logging.info("Evaluating model...")
        
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
        
        logging.info("Feature Importance:")
        for _, row in self.feature_importance.iterrows():
            logging.info(f"   {row['feature']}: {row['importance']:.3f}")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_classes': len(self.h3_encoder.classes_)
        }
    
    def save_model(self, model_dir: str = None):
        """
        Save trained model and encoder.
        
        Args:
            model_dir: Directory to save model files
        """
        if model_dir is None:
            model_dir = f'{self.data_dir}/models/final_models'
        
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'simple_h3_predictor.pkl')
        encoder_path = os.path.join(model_dir, 'h3_label_encoder.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.h3_encoder, encoder_path)
        
        logging.info(f"Model saved to: {model_path}")
        logging.info(f"Encoder saved to: {encoder_path}")
        
        return model_path, encoder_path
    
    def test_prediction(self, X_test: pd.DataFrame, y_test: np.ndarray):
        """
        Test a sample prediction and display results.
        
        Args:
            X_test: Test features
            y_test: Test targets
        """
        logging.info("Testing sample prediction...")
        
        # Take first test sample
        sample_input = X_test.iloc[0:1]
        predicted_encoded = self.model.predict(sample_input)[0]
        predicted_h3 = self.h3_encoder.inverse_transform([predicted_encoded])[0]
        actual_h3 = self.h3_encoder.inverse_transform([y_test[0]])[0]
        
        logging.info(f"Sample prediction:")
        logging.info(f"   Input: speed={sample_input['current_speed'].iloc[0]:.1f}, "
                    f"heading={sample_input['current_heading'].iloc[0]:.1f}")
        logging.info(f"   Predicted H3: {predicted_h3}")
        logging.info(f"   Actual H3: {actual_h3}")
        logging.info(f"   Correct: {'✅' if predicted_h3 == actual_h3 else '❌'}")
    
    def train_simple_h3_predictor(self, training_path: str = None, 
                                 output_dir: str = None) -> Tuple[RandomForestClassifier, LabelEncoder, Dict]:
        """
        Complete training pipeline for simple H3 predictor.
        
        Args:
            training_path: Path to training data
            output_dir: Directory to save model
            
        Returns:
            Tuple of (model, encoder, metrics)
        """
        logging.info("🤖 Training Simple H3 Cell Predictor...")
        
        # Load and prepare data
        training_df = self.load_training_data(training_path)
        X, y = self.prepare_features_and_target(training_df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(X_train, X_test, y_train, y_test)
        
        # Save model
        self.save_model(output_dir)
        
        # Test prediction
        self.test_prediction(X_test, y_test)
        
        logging.info(f"🎉 SUCCESS! H3 Predictor trained with {metrics['test_accuracy']:.1%} accuracy!")
        logging.info("🚀 You now have a working ML model that predicts vessel movements!")
        
        return self.model, self.h3_encoder, metrics
    
    def load_trained_model(self, model_dir: str = None) -> Tuple[RandomForestClassifier, LabelEncoder]:
        """
        Load a previously trained model.
        
        Args:
            model_dir: Directory containing saved model files
            
        Returns:
            Tuple of (model, encoder)
        """
        if model_dir is None:
            model_dir = f'{self.data_dir}/models/final_models'
        
        model_path = os.path.join(model_dir, 'simple_h3_predictor.pkl')
        encoder_path = os.path.join(model_dir, 'h3_label_encoder.pkl')
        
        self.model = joblib.load(model_path)
        self.h3_encoder = joblib.load(encoder_path)
        
        logging.info(f"Model loaded from: {model_path}")
        logging.info(f"Encoder loaded from: {encoder_path}")
        
        return self.model, self.h3_encoder
