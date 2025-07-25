"""
Tests for model implementations.
"""

import pytest
import torch
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.base_model import BaseTimeSeriesModel
from src.models.tft_model import TFTModel
from src.models.nbeats_model import NBeatsModel


class TestBaseTimeSeriesModel:
    """Test cases for BaseTimeSeriesModel abstract class."""
    
    def test_abstract_methods(self):
        """Test that BaseTimeSeriesModel cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseTimeSeriesModel({})


class TestTFTModel:
    """Test cases for TFTModel class."""
    
    def test_init(self):
        """Test TFTModel initialization."""
        config = {
            'max_prediction_length': 7,
            'max_encoder_length': 30,
            'learning_rate': 0.01,
            'hidden_size': 64,
            'attention_head_size': 2,
            'dropout': 0.1,
            'batch_size': 32,
            'max_epochs': 10
        }
        
        model = TFTModel(config)
        
        assert model.max_prediction_length == 7
        assert model.max_encoder_length == 30
        assert model.learning_rate == 0.01
        assert model.hidden_size == 64
        assert model.attention_head_size == 2
        assert model.dropout == 0.1
        assert model.batch_size == 32
        assert model.max_epochs == 10
    
    def test_create_datasets(self):
        """Test dataset creation."""
        config = {
            'max_prediction_length': 7,
            'max_encoder_length': 30,
            'days_left_for_testing': 7,
            'batch_size': 32
        }
        
        # Create test data
        test_data = pd.DataFrame({
            'time_idx': list(range(100)) * 2,
            'value': np.random.randn(200),
            'GroupIDS': ['A'] * 100 + ['B'] * 100,
            'month': np.random.randint(1, 13, 200),
            'day_of_week': np.random.randint(1, 8, 200)
        })
        
        model = TFTModel(config)
        training, validation = model._create_datasets(test_data)
        
        assert training is not None
        assert validation is not None
        assert model.training_dataset is not None
    
    @patch('src.models.tft_model.TemporalFusionTransformer')
    @patch('src.models.tft_model.pl.Trainer')
    def test_fit(self, mock_trainer, mock_tft):
        """Test model fitting."""
        config = {
            'max_prediction_length': 7,
            'max_encoder_length': 30,
            'days_left_for_testing': 7,
            'batch_size': 32,
            'max_epochs': 5,
            'num_workers': 0
        }
        
        # Create test data
        test_data = pd.DataFrame({
            'time_idx': list(range(50)) * 2,
            'value': np.random.randn(100),
            'GroupIDS': ['A'] * 50 + ['B'] * 50,
            'month': np.random.randint(1, 13, 100),
            'day_of_week': np.random.randint(1, 8, 100)
        })
        
        # Mock objects
        mock_model_instance = MagicMock()
        mock_tft.from_dataset.return_value = mock_model_instance
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        model = TFTModel(config)
        result = model.fit(df_data=test_data)
        
        assert result == model
        assert mock_tft.from_dataset.called
        assert mock_trainer_instance.fit.called
    
    def test_save_without_training(self):
        """Test saving model without training should raise error."""
        config = {'max_prediction_length': 7}
        model = TFTModel(config)
        
        with pytest.raises(ValueError, match="Model must be trained before saving"):
            with tempfile.NamedTemporaryFile() as temp_file:
                model.save(temp_file.name)


class TestNBeatsModel:
    """Test cases for NBeatsModel class."""
    
    def test_init(self):
        """Test NBeatsModel initialization."""
        config = {
            'max_prediction_length': 7,
            'max_encoder_length': 30,
            'learning_rate': 0.01,
            'widths': [32, 512],
            'backcast_loss_ratio': 0.0,
            'batch_size': 32,
            'max_epochs': 10
        }
        
        model = NBeatsModel(config)
        
        assert model.max_prediction_length == 7
        assert model.max_encoder_length == 30
        assert model.learning_rate == 0.01
        assert model.widths == [32, 512]
        assert model.backcast_loss_ratio == 0.0
        assert model.batch_size == 32
        assert model.max_epochs == 10
    
    def test_create_datasets(self):
        """Test dataset creation."""
        config = {
            'max_prediction_length': 7,
            'max_encoder_length': 30,
            'days_left_for_testing': 7,
            'batch_size': 32
        }
        
        # Create test data
        test_data = pd.DataFrame({
            'time_idx': list(range(100)) * 2,
            'value': np.random.randn(200),
            'GroupIDS': ['A'] * 100 + ['B'] * 100,
            'month': np.random.randint(1, 13, 200),
            'day_of_week': np.random.randint(1, 8, 200)
        })
        
        model = NBeatsModel(config)
        training, validation = model._create_datasets(test_data)
        
        assert training is not None
        assert validation is not None
        assert model.training_dataset is not None
    
    @patch('src.models.nbeats_model.NBeats')
    @patch('src.models.nbeats_model.pl.Trainer')
    def test_fit(self, mock_trainer, mock_nbeats):
        """Test model fitting."""
        config = {
            'max_prediction_length': 7,
            'max_encoder_length': 30,
            'days_left_for_testing': 7,
            'batch_size': 32,
            'max_epochs': 5,
            'num_workers': 0
        }
        
        # Create test data
        test_data = pd.DataFrame({
            'time_idx': list(range(50)) * 2,
            'value': np.random.randn(100),
            'GroupIDS': ['A'] * 50 + ['B'] * 50,
            'month': np.random.randint(1, 13, 100),
            'day_of_week': np.random.randint(1, 8, 100)
        })
        
        # Mock objects
        mock_model_instance = MagicMock()
        mock_nbeats.from_dataset.return_value = mock_model_instance
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        model = NBeatsModel(config)
        result = model.fit(df_data=test_data)
        
        assert result == model
        assert mock_nbeats.from_dataset.called
        assert mock_trainer_instance.fit.called
    
    def test_save_without_training(self):
        """Test saving model without training should raise error."""
        config = {'max_prediction_length': 7}
        model = NBeatsModel(config)
        
        with pytest.raises(ValueError, match="Model must be trained before saving"):
            with tempfile.NamedTemporaryFile() as temp_file:
                model.save(temp_file.name)
    
    def test_predict_without_training(self):
        """Test prediction without training should raise error."""
        config = {'max_prediction_length': 7}
        model = NBeatsModel(config)
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            mock_dataloader = MagicMock()
            model.predict(mock_dataloader)
    
    def test_evaluate_without_training(self):
        """Test evaluation without training should raise error."""
        config = {'max_prediction_length': 7}
        model = NBeatsModel(config)
        
        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            mock_dataloader = MagicMock()
            model.evaluate(mock_dataloader)


if __name__ == "__main__":
    pytest.main([__file__])
