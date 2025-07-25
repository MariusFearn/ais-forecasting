"""
Tests for feature engineering functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.features.geo_features import GeoFeatureEngineer
from src.features.time_features import TimeFeatureEngineer


class TestGeoFeatureEngineer:
    """Test cases for GeoFeatureEngineer class."""
    
    def test_init(self):
        """Test GeoFeatureEngineer initialization."""
        engineer = GeoFeatureEngineer(h3_resolution=8)
        assert engineer.h3_resolution == 8
    
    @patch('h3.geo_to_h3')
    def test_create_h3_cells(self, mock_geo_to_h3):
        """Test H3 cell creation."""
        mock_geo_to_h3.return_value = '881f1d4a07fffff'
        
        test_data = pd.DataFrame({
            'lat': [60.1699, 59.9139],
            'lon': [24.9384, 10.7522]
        })
        
        engineer = GeoFeatureEngineer(h3_resolution=8)
        result = engineer.create_h3_cells(test_data)
        
        assert 'h3_cell' in result.columns
        assert len(result) == len(test_data)
        assert mock_geo_to_h3.call_count == len(test_data)
    
    @patch('h3.h3_to_geo')
    def test_create_h3_center_coordinates(self, mock_h3_to_geo):
        """Test H3 center coordinate creation."""
        mock_h3_to_geo.return_value = (60.1699, 24.9384)
        
        test_data = pd.DataFrame({
            'h3_cell': ['881f1d4a07fffff', '881f1d4a08fffff']
        })
        
        engineer = GeoFeatureEngineer()
        result = engineer.create_h3_center_coordinates(test_data)
        
        assert 'h3_center_lat' in result.columns
        assert 'h3_center_lon' in result.columns
        assert len(result) == len(test_data)
    
    def test_calculate_distance_features(self):
        """Test distance feature calculation."""
        test_data = pd.DataFrame({
            'lat': [60.1699, 60.1700],
            'lon': [24.9384, 24.9385],
            'vessel_id': ['V001', 'V001']
        })
        test_data = test_data.sort_values(['vessel_id', 'lat', 'lon']).reset_index(drop=True)
        
        engineer = GeoFeatureEngineer()
        result = engineer.calculate_distance_features(test_data)
        
        assert 'distance_from_previous' in result.columns
        assert pd.isna(result['distance_from_previous'].iloc[0])  # First point has no previous
        assert result['distance_from_previous'].iloc[1] > 0  # Second point has distance
    
    def test_create_speed_features(self):
        """Test speed feature calculation."""
        # Create test data with timestamps
        timestamps = pd.date_range('2023-01-01 00:00:00', periods=3, freq='H')
        test_data = pd.DataFrame({
            'timestamp': timestamps,
            'lat': [60.1699, 60.1710, 60.1720],
            'lon': [24.9384, 24.9390, 24.9395],
            'vessel_id': ['V001', 'V001', 'V001']
        })
        
        engineer = GeoFeatureEngineer()
        result = engineer.create_speed_features(test_data)
        
        assert 'calculated_speed' in result.columns
        assert pd.isna(result['calculated_speed'].iloc[0])  # First point has no speed
        assert result['calculated_speed'].iloc[1] > 0  # Second point has speed


class TestTimeFeatureEngineer:
    """Test cases for TimeFeatureEngineer class."""
    
    def test_init(self):
        """Test TimeFeatureEngineer initialization."""
        engineer = TimeFeatureEngineer()
        assert engineer is not None
    
    def test_create_basic_time_features(self):
        """Test basic time feature creation."""
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D')
        })
        
        engineer = TimeFeatureEngineer()
        result = engineer.create_basic_time_features(test_data, 'timestamp')
        
        expected_columns = ['year', 'month', 'day', 'day_of_week', 'hour', 'minute']
        for col in expected_columns:
            assert col in result.columns
        
        # Check some values
        assert result['month'].iloc[0] == 1  # January
        assert result['day'].iloc[0] == 1    # First day
        assert result['year'].iloc[0] == 2023
    
    def test_create_cyclical_features(self):
        """Test cyclical feature creation."""
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=24, freq='H')
        })
        
        engineer = TimeFeatureEngineer()
        result = engineer.create_cyclical_features(test_data, 'timestamp')
        
        cyclical_columns = [
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos'
        ]
        
        for col in cyclical_columns:
            assert col in result.columns
        
        # Check that sin/cos values are in correct range
        for col in cyclical_columns:
            assert result[col].between(-1, 1).all()
    
    def test_create_lag_features(self):
        """Test lag feature creation."""
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': range(10),
            'group_id': ['A'] * 10
        })
        
        engineer = TimeFeatureEngineer()
        result = engineer.create_lag_features(
            test_data, 
            target_col='value',
            group_cols=['group_id'],
            lags=[1, 2, 3]
        )
        
        assert 'value_lag_1' in result.columns
        assert 'value_lag_2' in result.columns
        assert 'value_lag_3' in result.columns
        
        # Check lag values
        assert result['value_lag_1'].iloc[1] == 0  # Second row, lag 1 = first row value
        assert result['value_lag_2'].iloc[2] == 0  # Third row, lag 2 = first row value
    
    def test_create_rolling_features(self):
        """Test rolling feature creation."""
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
            'value': range(20),
            'group_id': ['A'] * 20
        })
        
        engineer = TimeFeatureEngineer()
        result = engineer.create_rolling_features(
            test_data,
            target_col='value',
            group_cols=['group_id'],
            windows=[3, 7],
            features=['mean', 'std']
        )
        
        expected_columns = [
            'value_rolling_3_mean', 'value_rolling_3_std',
            'value_rolling_7_mean', 'value_rolling_7_std'
        ]
        
        for col in expected_columns:
            assert col in result.columns
        
        # Check that rolling means are calculated correctly
        # For window=3, the 3rd row (index 2) should have mean of [0,1,2] = 1.0
        assert result['value_rolling_3_mean'].iloc[2] == 1.0
    
    def test_create_seasonal_features(self):
        """Test seasonal feature creation."""
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=365, freq='D')
        })
        
        engineer = TimeFeatureEngineer()
        result = engineer.create_seasonal_features(test_data, 'timestamp')
        
        seasonal_columns = ['is_weekend', 'is_holiday', 'season']
        for col in seasonal_columns:
            assert col in result.columns
        
        # Check weekend detection
        # 2023-01-01 was a Sunday
        assert result['is_weekend'].iloc[0] == True
        
        # Check season assignment
        seasons = result['season'].unique()
        expected_seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        for season in seasons:
            assert season in expected_seasons


if __name__ == "__main__":
    pytest.main([__file__])
