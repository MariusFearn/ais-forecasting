"""
Tests for data processing functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.loader import AISDataLoader
from src.data.preprocessing import AISDataPreprocessor


class TestAISDataLoader:
    """Test cases for AISDataLoader class."""
    
    def test_init(self):
        """Test AISDataLoader initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = AISDataLoader(temp_dir)
            assert loader.data_dir == temp_dir
            assert os.path.exists(loader.raw_dir)
            assert os.path.exists(loader.processed_dir)
    
    def test_save_and_load_processed_data(self):
        """Test saving and loading processed data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = AISDataLoader(temp_dir)
            
            # Create test data
            test_data = pd.DataFrame({
                'time_idx': range(10),
                'value': np.random.randn(10),
                'group_id': ['A'] * 5 + ['B'] * 5
            })
            
            # Save data
            loader.save_processed_data(test_data, "test_data")
            
            # Load data
            loaded_data = loader.load_processed_data("test_data")
            
            # Compare
            pd.testing.assert_frame_equal(test_data, loaded_data)
    
    def test_load_raw_data_csv(self):
        """Test loading raw CSV data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = AISDataLoader(temp_dir)
            
            # Create test CSV file
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10),
                'vessel_id': ['V001'] * 10,
                'lat': np.random.uniform(-90, 90, 10),
                'lon': np.random.uniform(-180, 180, 10)
            })
            
            csv_path = os.path.join(loader.raw_dir, "test.csv")
            test_data.to_csv(csv_path, index=False)
            
            # Load data
            loaded_data = loader.load_raw_data(csv_path)
            
            # Compare (excluding index)
            pd.testing.assert_frame_equal(
                test_data.reset_index(drop=True), 
                loaded_data.reset_index(drop=True)
            )
    
    def test_create_time_series_dataset(self):
        """Test creating TimeSeriesDataSet."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = AISDataLoader(temp_dir)
            
            # Create test data
            test_data = pd.DataFrame({
                'time_idx': list(range(100)) * 2,
                'value': np.random.randn(200),
                'group_id': ['A'] * 100 + ['B'] * 100,
                'month': np.random.randint(1, 13, 200),
                'day_of_week': np.random.randint(1, 8, 200)
            })
            
            # Create datasets
            training, validation = loader.create_time_series_dataset(
                test_data,
                time_idx='time_idx',
                target='value',
                group_ids=['group_id'],
                max_encoder_length=30,
                max_prediction_length=7,
                time_varying_known_reals=['month', 'day_of_week'],
                time_varying_unknown_reals=['value']
            )
            
            assert training is not None
            assert validation is not None


class TestAISDataPreprocessor:
    """Test cases for AISDataPreprocessor class."""
    
    def test_validate_ais_data(self):
        """Test AIS data validation."""
        # Create test data with various issues
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'vessel_id': ['V001'] * 8 + [None, None],  # Missing vessel IDs
            'lat': list(np.random.uniform(-90, 90, 8)) + [91, -91],  # Invalid lat
            'lon': list(np.random.uniform(-180, 180, 8)) + [181, -181],  # Invalid lon
            'speed': list(np.random.uniform(0, 30, 8)) + [-1, 100]  # Invalid speed
        })
        
        preprocessor = AISDataPreprocessor()
        
        # Test validation (should return issues)
        issues = preprocessor.validate_ais_data(test_data)
        
        assert len(issues) > 0
        assert any('vessel_id' in issue for issue in issues)
        assert any('latitude' in issue for issue in issues)
        assert any('longitude' in issue for issue in issues)
    
    def test_clean_ais_data(self):
        """Test AIS data cleaning."""
        # Create test data with issues
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'vessel_id': ['V001'] * 8 + [None, None],
            'lat': list(np.random.uniform(-90, 90, 8)) + [91, -91],
            'lon': list(np.random.uniform(-180, 180, 8)) + [181, -181],
            'speed': list(np.random.uniform(0, 30, 8)) + [-1, 100]
        })
        
        preprocessor = AISDataPreprocessor()
        
        # Clean data
        cleaned_data = preprocessor.clean_ais_data(test_data)
        
        # Should have fewer rows after cleaning
        assert len(cleaned_data) < len(test_data)
        
        # Should have no missing vessel IDs
        assert cleaned_data['vessel_id'].notna().all()
        
        # Should have valid coordinates
        assert (cleaned_data['lat'].between(-90, 90)).all()
        assert (cleaned_data['lon'].between(-180, 180)).all()
    
    @patch('h3.geo_to_h3')
    def test_create_h3_features(self, mock_geo_to_h3):
        """Test H3 feature creation."""
        # Mock H3 function
        mock_geo_to_h3.return_value = '881f1d4a07fffff'
        
        test_data = pd.DataFrame({
            'lat': [60.1699, 59.9139],
            'lon': [24.9384, 10.7522]
        })
        
        preprocessor = AISDataPreprocessor()
        
        # Create H3 features
        result = preprocessor.create_h3_features(test_data, resolution=8)
        
        assert 'h3_cell' in result.columns
        assert mock_geo_to_h3.call_count == len(test_data)
    
    def test_create_time_features(self):
        """Test time feature creation."""
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D')
        })
        
        preprocessor = AISDataPreprocessor()
        
        # Create time features
        result = preprocessor.create_time_features(test_data, 'timestamp')
        
        expected_columns = ['year', 'month', 'day', 'day_of_week', 'hour', 'is_weekend']
        for col in expected_columns:
            assert col in result.columns
        
        # Check some values
        assert result['month'].iloc[0] == 1  # January
        assert result['day'].iloc[0] == 1    # First day
        assert result['year'].iloc[0] == 2023


if __name__ == "__main__":
    pytest.main([__file__])
