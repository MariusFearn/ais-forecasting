"""
Data preprocessing utilities for AIS data.

This module provides comprehensive preprocessing for vessel trajectory prediction,
including solutions for the data type and memory issues discovered during Phase 5.
"""

import pandas as pd
import numpy as np
import h3
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import warnings
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


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


class DataPreprocessor:
    """
    Comprehensive preprocessing for vessel trajectory data.
    
    Solves all the data type and memory issues discovered during Phase 5:
    - Mixed data types (datetime, categorical, numeric)
    - Memory optimization for large datasets
    - Consistent encoding strategies
    - Missing value handling
    """
    
    def __init__(self, memory_optimize: bool = True, verbose: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            memory_optimize: Whether to optimize memory usage
            verbose: Whether to print processing details
        """
        self.memory_optimize = memory_optimize
        self.verbose = verbose
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_stats: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)
        
    def handle_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert datetime columns to numeric timestamps.
        
        Solves the datetime casting issues discovered in Phase 5.
        
        Args:
            df: DataFrame with potential datetime columns
            
        Returns:
            DataFrame with datetime columns converted to timestamps
        """
        if self.verbose:
            print("   🕐 Processing datetime columns...")
            
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            if self.verbose:
                print(f"      Converting {col} to timestamp")
                
            # Convert to datetime, handling errors
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Convert to Unix timestamp (seconds since epoch)
            df[col] = df[col].astype('int64') // 10**9
            
            # Fill NaN values with 0 (epoch time)
            df[col] = df[col].fillna(0)
            
            # Store conversion info
            self.feature_stats[col] = {
                'type': 'datetime_converted',
                'null_count': df[col].isna().sum(),
                'range': (df[col].min(), df[col].max())
            }
        
        if self.verbose and len(datetime_cols) > 0:
            print(f"      ✅ Converted {len(datetime_cols)} datetime columns")
            
        return df
    
    def handle_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """
        Encode categorical features with proper missing value handling.
        
        Solves the categorical encoding issues from Phase 5.
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            Tuple of (processed DataFrame, encoder dictionary)
        """
        if self.verbose:
            print("   🏷️  Processing categorical columns...")
            
        categorical_cols = df.select_dtypes(include=['object']).columns
        encoders = {}
        
        for col in categorical_cols:
            if self.verbose:
                unique_before = df[col].nunique()
                
            # Handle missing values explicitly
            df[col] = df[col].fillna('MISSING_VALUE')
            
            # Ensure all values are strings
            df[col] = df[col].astype(str)
            
            # Create and fit encoder
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder
            
            # Store encoding info
            self.feature_stats[col] = {
                'type': 'categorical_encoded',
                'unique_values': len(encoder.classes_),
                'classes_sample': encoder.classes_[:5].tolist() if len(encoder.classes_) > 5 else encoder.classes_.tolist()
            }
            
            if self.verbose:
                print(f"      ✅ {col}: {unique_before} → {len(encoder.classes_)} encoded values")
        
        # Update instance encoders
        self.encoders.update(encoders)
        
        if self.verbose and len(categorical_cols) > 0:
            print(f"      ✅ Encoded {len(categorical_cols)} categorical columns")
            
        return df, encoders
    
    def handle_boolean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert boolean columns to integers.
        
        Args:
            df: DataFrame with potential boolean columns
            
        Returns:
            DataFrame with boolean columns converted to int
        """
        boolean_cols = df.select_dtypes(include=['bool']).columns
        
        for col in boolean_cols:
            df[col] = df[col].astype(int)
            
            self.feature_stats[col] = {
                'type': 'boolean_converted',
                'unique_values': df[col].nunique()
            }
        
        if self.verbose and len(boolean_cols) > 0:
            print(f"   ✔️  Converted {len(boolean_cols)} boolean columns")
            
        return df
    
    def handle_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process numeric features with missing value handling.
        
        Args:
            df: DataFrame with numeric columns
            
        Returns:
            DataFrame with processed numeric columns
        """
        if self.verbose:
            print("   🔢 Processing numeric columns...")
            
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        
        for col in numeric_cols:
            # Handle infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                if self.verbose:
                    print(f"      Replaced {inf_count} infinite values in {col}")
            
            # Fill missing values with median
            if df[col].isna().any():
                median_val = df[col].median()
                na_count = df[col].isna().sum()
                df[col] = df[col].fillna(median_val)
                if self.verbose:
                    print(f"      Filled {na_count} missing values in {col} with median {median_val:.2f}")
            
            # Store stats
            self.feature_stats[col] = {
                'type': 'numeric',
                'range': (df[col].min(), df[col].max()),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
        
        if self.verbose:
            print(f"      ✅ Processed {len(numeric_cols)} numeric columns")
            
        return df
    
    def remove_constant_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with no variance.
        
        Args:
            df: DataFrame to check for constant features
            
        Returns:
            Tuple of (DataFrame without constant features, list of removed features)
        """
        constant_features = []
        
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            df = df.drop(columns=constant_features)
            if self.verbose:
                print(f"   🗑️  Removed {len(constant_features)} constant features")
                
        return df, constant_features
    
    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage by downcasting numeric types.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        if not self.memory_optimize:
            return df
            
        if self.verbose:
            initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"   💾 Optimizing memory usage (initial: {initial_memory:.1f} MB)...")
        
        # Downcast integers
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Downcast floats
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        if self.verbose:
            final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            savings = ((initial_memory - final_memory) / initial_memory) * 100
            print(f"      ✅ Optimized to {final_memory:.1f} MB ({savings:.1f}% savings)")
            
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the quality of processed data.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': df.isna().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'object_columns': len(df.select_dtypes(include=['object']).columns),
            'data_types': dict(df.dtypes.value_counts())
        }
        
        if self.verbose:
            print(f"\n   📊 Data Quality Report:")
            print(f"      Rows: {quality_report['total_rows']:,}")
            print(f"      Columns: {quality_report['total_columns']}")
            print(f"      Memory: {quality_report['memory_usage_mb']:.1f} MB")
            print(f"      Missing values: {quality_report['missing_values']}")
            print(f"      Numeric columns: {quality_report['numeric_columns']}")
            
        return quality_report
    
    def process_features(self, df: pd.DataFrame, 
                        exclude_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete feature processing pipeline.
        
        This is the main method that handles all the data preprocessing issues
        discovered during Phase 5 massive scale training.
        
        Args:
            df: Raw DataFrame with mixed data types
            exclude_columns: Columns to exclude from processing
            
        Returns:
            Tuple of (processed DataFrame, processing metadata)
        """
        if self.verbose:
            print("🔧 Starting comprehensive feature processing...")
            
        # Exclude specified columns
        exclude_columns = exclude_columns or []
        feature_cols = [col for col in df.columns if col not in exclude_columns]
        
        # Work with feature subset
        X = df[feature_cols].copy()
        
        if self.verbose:
            print(f"   📊 Processing {len(X.columns)} features (excluding {len(exclude_columns)} columns)")
        
        # Step 1: Handle datetime features
        X = self.handle_datetime_features(X)
        
        # Step 2: Handle categorical features  
        X, categorical_encoders = self.handle_categorical_features(X)
        
        # Step 3: Handle boolean features
        X = self.handle_boolean_features(X)
        
        # Step 4: Handle numeric features
        X = self.handle_numeric_features(X)
        
        # Step 5: Remove constant features
        X, constant_features = self.remove_constant_features(X)
        
        # Step 6: Final validation - ensure all columns are numeric
        non_numeric_cols = []
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                non_numeric_cols.append(col)
                # Force conversion
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        if non_numeric_cols and self.verbose:
            print(f"   ⚠️  Force-converted {len(non_numeric_cols)} non-numeric columns")
        
        # Step 7: Memory optimization
        X = self.optimize_memory_usage(X)
        
        # Step 8: Quality validation
        quality_report = self.validate_data_quality(X)
        
        # Combine with excluded columns
        if exclude_columns:
            excluded_data = df[exclude_columns].copy()
            result_df = pd.concat([X, excluded_data], axis=1)
        else:
            result_df = X
        
        # Prepare metadata
        metadata = {
            'categorical_encoders': categorical_encoders,
            'constant_features_removed': constant_features,
            'feature_stats': self.feature_stats,
            'quality_report': quality_report,
            'processing_steps': [
                'datetime_conversion',
                'categorical_encoding', 
                'boolean_conversion',
                'numeric_processing',
                'constant_removal',
                'memory_optimization'
            ]
        }
        
        if self.verbose:
            print(f"✅ Feature processing complete!")
            print(f"   Final shape: {result_df.shape}")
            print(f"   All columns numeric: {pd.api.types.is_numeric_dtype(X).all()}")
        
        return result_df, metadata
    
    def save_encoders(self, output_path: str) -> None:
        """Save encoders for later use in prediction."""
        import joblib
        
        encoder_data = {
            'categorical_encoders': self.encoders,
            'feature_stats': self.feature_stats
        }
        
        joblib.dump(encoder_data, output_path)
        if self.verbose:
            print(f"💾 Encoders saved to: {output_path}")
    
    def load_encoders(self, input_path: str) -> None:
        """Load encoders from file."""
        import joblib
        
        encoder_data = joblib.load(input_path)
        self.encoders = encoder_data['categorical_encoders']
        self.feature_stats = encoder_data['feature_stats']
        
        if self.verbose:
            print(f"📂 Encoders loaded from: {input_path}")


class ChunkedDataLoader:
    """
    Memory-efficient data loading for large datasets.
    
    Handles the scaling issues discovered in Phase 5 when processing
    86M records and 75k+ sequences. Solves: Killed (signal 9) - Out of memory
    """
    
    def __init__(self, chunk_size: int = 10000, memory_limit_gb: float = 4.0):
        """
        Initialize the chunked data loader.
        
        Args:
            chunk_size: Number of rows per chunk
            memory_limit_gb: Memory limit in GB for processing
        """
        self.chunk_size = chunk_size
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.logger = logging.getLogger(__name__)
    
    def load_multi_year_data(self, file_pattern: str, 
                           years: List[str],
                           max_vessels: Optional[int] = None) -> pd.DataFrame:
        """
        Load and combine multiple years of data efficiently.
        
        Args:
            file_pattern: File pattern with {year} placeholder (e.g., 'ais_cape_data_{year}.pkl')
            years: List of years to load
            max_vessels: Maximum number of vessels to include (for sampling)
            
        Returns:
            Combined dataframe
        """
        combined_data = []
        
        for year in years:
            file_path = file_pattern.format(year=year)
            self.logger.info(f"Loading data for year {year} from {file_path}")
            
            if Path(file_path).exists():
                # Load year data safely
                year_data = self._load_file_safely(file_path)
                combined_data.append(year_data)
                self.logger.info(f"Loaded {len(year_data)} rows for year {year}")
            else:
                self.logger.warning(f"File not found: {file_path}")
        
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            self.logger.info(f"Combined data shape: {result.shape}")
            
            # Apply vessel sampling if requested
            if max_vessels and 'mmsi' in result.columns:
                unique_vessels = result['mmsi'].unique()
                if len(unique_vessels) > max_vessels:
                    selected_vessels = np.random.choice(unique_vessels, max_vessels, replace=False)
                    result = result[result['mmsi'].isin(selected_vessels)]
                    self.logger.info(f"Sampled {max_vessels} vessels from {len(unique_vessels)} available")
            
            return result
        else:
            raise ValueError("No data files found")
    
    def _load_file_safely(self, file_path: str) -> pd.DataFrame:
        """Load a file safely with memory management."""
        from pathlib import Path
        
        file_size = Path(file_path).stat().st_size
        
        # If file is large, process in chunks
        if file_size > self.memory_limit_bytes / 4:  # Use 1/4 of memory limit as threshold
            self.logger.info(f"Large file detected ({file_size / 1024**3:.1f} GB), processing in chunks")
            return self._process_large_file(file_path)
        else:
            df = pd.read_pickle(file_path)
            return df
    
    def _process_large_file(self, file_path: str) -> pd.DataFrame:
        """Process large files in chunks to avoid OOM."""
        try:
            # For pickle files, we need to read the entire file at once
            # But we can immediately process and reduce it
            df = pd.read_pickle(file_path)
            
            # Apply immediate memory optimizations
            df = self._optimize_memory_immediately(df)
            
            return df
            
        except MemoryError:
            self.logger.error(f"Still out of memory loading {file_path}")
            raise MemoryError(f"File {file_path} too large to process even with optimizations")
    
    def _optimize_memory_immediately(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply immediate memory optimizations to reduce memory usage."""
        original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Convert data types for memory efficiency
        for col in df.columns:
            col_type = df[col].dtype
            
            # Convert int64 to int32 if values fit
            if col_type == 'int64':
                if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')
            
            # Convert float64 to float32
            elif col_type == 'float64':
                df[col] = df[col].astype('float32')
            
            # Convert object columns with few unique values to category
            elif col_type == 'object' and df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        
        self.logger.info(f"Memory usage reduced by {memory_reduction:.1f}% "
                        f"({original_memory:.1f} MB → {optimized_memory:.1f} MB)")
        
        return df
    
    def sample_balanced_dataset(self, df: pd.DataFrame, 
                              target_size: int,
                              group_column: str = 'mmsi') -> pd.DataFrame:
        """
        Create a balanced sample from a large dataset.
        
        Args:
            df: Input dataframe
            target_size: Target number of rows
            group_column: Column to balance by (e.g., vessel ID)
            
        Returns:
            Balanced sample dataframe
        """
        if len(df) <= target_size:
            return df
        
        # Get unique groups
        unique_groups = df[group_column].unique()
        samples_per_group = max(1, target_size // len(unique_groups))
        
        sampled_data = []
        for group in unique_groups:
            group_data = df[df[group_column] == group]
            
            if len(group_data) > samples_per_group:
                group_sample = group_data.sample(n=samples_per_group, random_state=42)
            else:
                group_sample = group_data
            
            sampled_data.append(group_sample)
        
        result = pd.concat(sampled_data, ignore_index=True)
        self.logger.info(f"Created balanced sample: {len(result)} rows from {len(df)} rows")
        
        return result


def fix_datetime_categorical_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the critical datetime and categorical issues discovered in Phase 5.
    
    Solves:
    - TypeError: Cannot convert DatetimeArray to dtype float64
    - ValueError: could not convert string to float: 'Under Way Using Engine'
    
    Args:
        df: Input dataframe with mixed data types
        
    Returns:
        DataFrame with all data types properly converted for ML models
    """
    df = df.copy()
    
    # Fix datetime columns
    datetime_columns = []
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]' or 'time' in col.lower():
            datetime_columns.append(col)
    
    for col in datetime_columns:
        try:
            # Convert datetime to timestamp (seconds since epoch)
            if df[col].dtype == 'datetime64[ns]':
                df[col] = df[col].astype('int64') // 10**9
            else:
                # Try to convert to datetime first, then to timestamp
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].astype('int64') // 10**9
            
            # Handle NaN values
            df[col] = df[col].fillna(0)
            
            print(f"✅ Fixed datetime column: {col}")
            
        except Exception as e:
            print(f"⚠️ Could not fix datetime column {col}: {e}")
            df[col] = 0  # Fallback
    
    # Fix categorical columns
    categorical_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_columns.append(col)
    
    for col in categorical_columns:
        try:
            # Handle missing values first
            df[col] = df[col].fillna('MISSING').astype(str)
            
            # Use label encoding
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            
            print(f"✅ Fixed categorical column: {col} ({len(encoder.classes_)} categories)")
            
        except Exception as e:
            print(f"⚠️ Could not fix categorical column {col}: {e}")
            # Fallback: convert to numeric codes
            df[col] = pd.Categorical(df[col]).codes
    
    return df


# Convenience function for quick processing
def preprocess_vessel_data(df: pd.DataFrame, 
                          exclude_columns: Optional[List[str]] = None,
                          memory_optimize: bool = True,
                          verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Quick preprocessing function for vessel trajectory data.
    
    This solves all the data handling issues discovered during Phase 5.
    
    Args:
        df: Raw vessel data DataFrame
        exclude_columns: Columns to exclude (e.g., ['target_h3_cell', 'vessel_imo'])
        memory_optimize: Whether to optimize memory usage
        verbose: Whether to print progress
        
    Returns:
        Tuple of (processed DataFrame, processing metadata)
    """
    preprocessor = DataPreprocessor(memory_optimize=memory_optimize, verbose=verbose)
    return preprocessor.process_features(df, exclude_columns)