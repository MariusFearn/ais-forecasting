"""
Data preprocessing utilities for AIS data.
"""

import pandas as pd
import numpy as np
import h3
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import warnings


class AISDataPreprocessor:
    """
    Class for preprocessing AIS data with validation and cleaning capabilities.
    """
    
    def __init__(self):
        """Initialize the AIS data preprocessor."""
        self.logger = logging.getLogger(__name__)
    
    def validate_ais_data(self, df: pd.DataFrame) -> List[str]:
        """
        Validate AIS data and return list of issues found.
        
        Args:
            df: DataFrame with AIS data
            
        Returns:
            List[str]: List of validation issues
        """
        issues = []
        
        # Check for required columns
        required_cols = ['timestamp', 'vessel_id', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        if not missing_cols:  # Only check data quality if required columns exist
            # Check for missing values
            for col in required_cols:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    missing_pct = (missing_count / len(df)) * 100
                    issues.append(f"Column '{col}' has {missing_count} missing values ({missing_pct:.1f}%)")
            
            # Check latitude bounds
            if 'lat' in df.columns:
                invalid_lat = ((df['lat'] < -90) | (df['lat'] > 90)).sum()
                if invalid_lat > 0:
                    issues.append(f"Found {invalid_lat} invalid latitude values (outside -90 to 90 range)")
            
            # Check longitude bounds
            if 'lon' in df.columns:
                invalid_lon = ((df['lon'] < -180) | (df['lon'] > 180)).sum()
                if invalid_lon > 0:
                    issues.append(f"Found {invalid_lon} invalid longitude values (outside -180 to 180 range)")
            
            # Check for duplicate records
            if all(col in df.columns for col in ['timestamp', 'vessel_id']):
                duplicates = df.duplicated(subset=['timestamp', 'vessel_id']).sum()
                if duplicates > 0:
                    issues.append(f"Found {duplicates} duplicate records based on timestamp and vessel_id")
            
            # Check speed if available
            if 'speed' in df.columns:
                invalid_speed = ((df['speed'] < 0) | (df['speed'] > 50)).sum()  # Reasonable speed limits
                if invalid_speed > 0:
                    issues.append(f"Found {invalid_speed} potentially invalid speed values (< 0 or > 50 knots)")
            
            # Check timestamp format
            if 'timestamp' in df.columns:
                try:
                    pd.to_datetime(df['timestamp'])
                except Exception:
                    issues.append("Timestamp column cannot be converted to datetime format")
        
        return issues
    
    def clean_ais_data(self, df: pd.DataFrame, 
                      remove_outliers: bool = True,
                      speed_threshold: float = 50.0) -> pd.DataFrame:
        """
        Clean AIS data by removing invalid records and outliers.
        
        Args:
            df: DataFrame with AIS data
            remove_outliers: Whether to remove speed outliers
            speed_threshold: Maximum reasonable speed in knots
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        # Convert timestamp to datetime
        if 'timestamp' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
        
        # Remove records with missing critical fields
        critical_fields = ['timestamp', 'vessel_id', 'lat', 'lon']
        for field in critical_fields:
            if field in df_clean.columns:
                before_count = len(df_clean)
                df_clean = df_clean.dropna(subset=[field])
                removed = before_count - len(df_clean)
                if removed > 0:
                    self.logger.info(f"Removed {removed} records with missing {field}")
        
        # Remove records with invalid coordinates
        if 'lat' in df_clean.columns and 'lon' in df_clean.columns:
            before_count = len(df_clean)
            df_clean = df_clean[
                (df_clean['lat'] >= -90) & (df_clean['lat'] <= 90) &
                (df_clean['lon'] >= -180) & (df_clean['lon'] <= 180)
            ]
            removed = before_count - len(df_clean)
            if removed > 0:
                self.logger.info(f"Removed {removed} records with invalid coordinates")
        
        # Remove speed outliers if requested
        if remove_outliers and 'speed' in df_clean.columns:
            before_count = len(df_clean)
            df_clean = df_clean[
                (df_clean['speed'] >= 0) & (df_clean['speed'] <= speed_threshold)
            ]
            removed = before_count - len(df_clean)
            if removed > 0:
                self.logger.info(f"Removed {removed} records with invalid speed (> {speed_threshold} knots)")
        
        # Remove duplicate records
        if all(col in df_clean.columns for col in ['timestamp', 'vessel_id']):
            before_count = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=['timestamp', 'vessel_id'])
            removed = before_count - len(df_clean)
            if removed > 0:
                self.logger.info(f"Removed {removed} duplicate records")
        
        # Sort by vessel and timestamp
        if all(col in df_clean.columns for col in ['vessel_id', 'timestamp']):
            df_clean = df_clean.sort_values(['vessel_id', 'timestamp']).reset_index(drop=True)
        
        final_count = len(df_clean)
        removal_pct = ((initial_count - final_count) / initial_count) * 100
        self.logger.info(f"Data cleaning complete: {initial_count} -> {final_count} records ({removal_pct:.1f}% removed)")
        
        return df_clean
    
    def create_h3_features(self, df: pd.DataFrame, 
                          resolution: int = 8,
                          lat_col: str = 'lat',
                          lon_col: str = 'lon') -> pd.DataFrame:
        """
        Create H3 geospatial features.
        
        Args:
            df: DataFrame with latitude and longitude
            resolution: H3 resolution level
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            
        Returns:
            pd.DataFrame: DataFrame with H3 features added
        """
        df = df.copy()
        
        def get_h3_cell(lat, lon):
            try:
                if pd.isna(lat) or pd.isna(lon):
                    return None
                return h3.geo_to_h3(lat, lon, resolution)
            except Exception:
                return None
        
        # Create H3 cell
        df['h3_cell'] = df.apply(lambda row: get_h3_cell(row[lat_col], row[lon_col]), axis=1)
        
        # Create H3 center coordinates
        def get_h3_center(h3_cell):
            try:
                if pd.isna(h3_cell):
                    return None, None
                lat, lon = h3.h3_to_geo(h3_cell)
                return lat, lon
            except Exception:
                return None, None
        
        centers = df['h3_cell'].apply(get_h3_center)
        df['h3_center_lat'] = [center[0] for center in centers]
        df['h3_center_lon'] = [center[1] for center in centers]
        
        return df
    
    def create_time_features(self, df: pd.DataFrame, 
                           timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Create basic time features.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            pd.DataFrame: DataFrame with time features added
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if df[timestamp_col].dtype == 'object':
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract time components
        df['year'] = df[timestamp_col].dt.year
        df['month'] = df[timestamp_col].dt.month
        df['day'] = df[timestamp_col].dt.day
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['hour'] = df[timestamp_col].dt.hour
        
        # Create cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def create_time_index(self, df: pd.DataFrame,
                         timestamp_col: str = 'timestamp',
                         freq: str = 'H') -> pd.DataFrame:
        """
        Create a sequential time index for forecasting models.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            freq: Frequency for time index ('H' for hourly, 'D' for daily)
            
        Returns:
            pd.DataFrame: DataFrame with time_idx column added
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if df[timestamp_col].dtype == 'object':
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Round timestamps to specified frequency
        if freq == 'H':
            df['timestamp_rounded'] = df[timestamp_col].dt.floor('H')
        elif freq == 'D':
            df['timestamp_rounded'] = df[timestamp_col].dt.floor('D')
        else:
            df['timestamp_rounded'] = df[timestamp_col]
        
        # Create sequential time index
        unique_times = df['timestamp_rounded'].unique()
        unique_times = pd.Series(unique_times).sort_values()
        time_mapping = {time: idx for idx, time in enumerate(unique_times)}
        
        df['time_idx'] = df['timestamp_rounded'].map(time_mapping)
        
        return df
    
    def aggregate_to_grid(self, df: pd.DataFrame,
                         h3_col: str = 'h3_cell',
                         time_col: str = 'time_idx',
                         value_cols: List[str] = None,
                         agg_functions: Dict[str, str] = None) -> pd.DataFrame:
        """
        Aggregate data to H3 grid cells over time.
        
        Args:
            df: DataFrame with H3 and time data
            h3_col: Name of H3 cell column
            time_col: Name of time index column
            value_cols: Columns to aggregate
            agg_functions: Dictionary mapping column names to aggregation functions
            
        Returns:
            pd.DataFrame: Aggregated DataFrame
        """
        if value_cols is None:
            value_cols = ['speed', 'calculated_speed'] if 'speed' in df.columns else []
        
        if agg_functions is None:
            agg_functions = {col: 'mean' for col in value_cols}
        
        df_agg = df.copy()
        
        # Group by H3 cell and time
        group_cols = [h3_col, time_col]
        
        # Add count of observations
        agg_dict = {'vessel_id': 'count'}
        agg_dict.update(agg_functions)
        
        df_result = df_agg.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Rename count column
        df_result = df_result.rename(columns={'vessel_id': 'vessel_count'})
        
        return df_result
    
    def handle_missing_values(self, df: pd.DataFrame,
                            strategy: str = 'interpolate',
                            columns: List[str] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with missing values
            strategy: Strategy for handling missing values ('interpolate', 'forward_fill', 'drop')
            columns: Columns to apply missing value handling to
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        df = df.copy()
        
        if columns is None:
            # Apply to numeric columns only
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    if strategy == 'interpolate':
                        df[col] = df[col].interpolate(method='linear')
                    elif strategy == 'forward_fill':
                        df[col] = df[col].fillna(method='ffill')
                    elif strategy == 'backward_fill':
                        df[col] = df[col].fillna(method='bfill')
                    elif strategy == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == 'median':
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == 'drop':
                        df = df.dropna(subset=[col])
                    
                    filled_count = missing_count - df[col].isna().sum()
                    self.logger.info(f"Filled {filled_count} missing values in column '{col}' using {strategy}")
        
        return df
    
    def normalize_features(self, df: pd.DataFrame,
                          columns: List[str] = None,
                          method: str = 'minmax') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize feature columns.
        
        Args:
            df: DataFrame with features to normalize
            columns: Columns to normalize (if None, normalize all numeric columns)
            method: Normalization method ('minmax', 'zscore')
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Normalized DataFrame and normalization parameters
        """
        df = df.copy()
        normalization_params = {}
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                if method == 'minmax':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    if max_val != min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                        normalization_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                    else:
                        normalization_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                        
                elif method == 'zscore':
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    
                    if std_val != 0:
                        df[col] = (df[col] - mean_val) / std_val
                        normalization_params[col] = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
                    else:
                        normalization_params[col] = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
        
        return df, normalization_params
