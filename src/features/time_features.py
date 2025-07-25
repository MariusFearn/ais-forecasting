"""
Temporal feature engineering for AIS data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import holidays


class TimeFeatureEngineer:
    """
    Class for creating temporal features from time series data.
    """
    
    def __init__(self, country_holidays: str = 'US'):
        """
        Initialize the temporal feature engineer.
        
        Args:
            country_holidays: Country code for holiday calendar
        """
        self.country_holidays = country_holidays
        self.logger = logging.getLogger(__name__)
        
        # Initialize holiday calendar
        try:
            self.holiday_calendar = holidays.country_holidays(country_holidays)
        except Exception:
            self.logger.warning(f"Could not load holidays for {country_holidays}, using US as default")
            self.holiday_calendar = holidays.US()
    
    def create_basic_time_features(self, df: pd.DataFrame, 
                                 timestamp_col: str) -> pd.DataFrame:
        """
        Create basic time features from timestamp.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            pd.DataFrame: DataFrame with basic time features added
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if df[timestamp_col].dtype == 'object':
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract basic time components
        df['year'] = df[timestamp_col].dt.year
        df['month'] = df[timestamp_col].dt.month
        df['day'] = df[timestamp_col].dt.day
        df['day_of_week'] = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6
        df['day_of_year'] = df[timestamp_col].dt.dayofyear
        df['week_of_year'] = df[timestamp_col].dt.isocalendar().week
        df['hour'] = df[timestamp_col].dt.hour
        df['minute'] = df[timestamp_col].dt.minute
        df['second'] = df[timestamp_col].dt.second
        
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame,
                               timestamp_col: str) -> pd.DataFrame:
        """
        Create cyclical features using sine and cosine transformations.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            pd.DataFrame: DataFrame with cyclical features added
        """
        df = df.copy()
        
        # Ensure we have basic time features
        if 'hour' not in df.columns:
            df = self.create_basic_time_features(df, timestamp_col)
        
        # Hour cyclical features (0-23)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week cyclical features (0-6)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month cyclical features (1-12)
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        
        # Day of year cyclical features (1-365/366)
        df['day_of_year_sin'] = np.sin(2 * np.pi * (df['day_of_year'] - 1) / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * (df['day_of_year'] - 1) / 365)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame,
                          target_col: str,
                          group_cols: List[str] = None,
                          lags: List[int] = None) -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column to create lags for
            group_cols: Columns to group by (e.g., vessel_id)
            lags: List of lag periods to create
            
        Returns:
            pd.DataFrame: DataFrame with lag features added
        """
        if lags is None:
            lags = [1, 2, 3, 7, 14]
        
        if group_cols is None:
            group_cols = []
        
        df = df.copy()
        
        # Sort by group columns and time index
        sort_cols = group_cols.copy()
        if 'timestamp' in df.columns:
            sort_cols.append('timestamp')
        elif 'time_idx' in df.columns:
            sort_cols.append('time_idx')
        
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        
        def create_lags_for_group(group_df):
            group_df = group_df.copy()
            for lag in lags:
                lag_col_name = f'{target_col}_lag_{lag}'
                group_df[lag_col_name] = group_df[target_col].shift(lag)
            return group_df
        
        if group_cols:
            df = df.groupby(group_cols).apply(create_lags_for_group).reset_index(drop=True)
        else:
            df = create_lags_for_group(df)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                              target_col: str,
                              group_cols: List[str] = None,
                              windows: List[int] = None,
                              features: List[str] = None) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            group_cols: Columns to group by (e.g., vessel_id)
            windows: List of window sizes
            features: List of rolling features to compute
            
        Returns:
            pd.DataFrame: DataFrame with rolling features added
        """
        if windows is None:
            windows = [3, 7, 14, 30]
        
        if features is None:
            features = ['mean', 'std', 'min', 'max']
        
        if group_cols is None:
            group_cols = []
        
        df = df.copy()
        
        # Sort by group columns and time index
        sort_cols = group_cols.copy()
        if 'timestamp' in df.columns:
            sort_cols.append('timestamp')
        elif 'time_idx' in df.columns:
            sort_cols.append('time_idx')
        
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        
        def create_rolling_for_group(group_df):
            group_df = group_df.copy()
            for window in windows:
                for feature in features:
                    col_name = f'{target_col}_rolling_{window}_{feature}'
                    
                    if feature == 'mean':
                        group_df[col_name] = group_df[target_col].rolling(window=window, min_periods=1).mean()
                    elif feature == 'std':
                        group_df[col_name] = group_df[target_col].rolling(window=window, min_periods=1).std()
                    elif feature == 'min':
                        group_df[col_name] = group_df[target_col].rolling(window=window, min_periods=1).min()
                    elif feature == 'max':
                        group_df[col_name] = group_df[target_col].rolling(window=window, min_periods=1).max()
                    elif feature == 'sum':
                        group_df[col_name] = group_df[target_col].rolling(window=window, min_periods=1).sum()
                    elif feature == 'median':
                        group_df[col_name] = group_df[target_col].rolling(window=window, min_periods=1).median()
                    elif feature == 'quantile_25':
                        group_df[col_name] = group_df[target_col].rolling(window=window, min_periods=1).quantile(0.25)
                    elif feature == 'quantile_75':
                        group_df[col_name] = group_df[target_col].rolling(window=window, min_periods=1).quantile(0.75)
            
            return group_df
        
        if group_cols:
            df = df.groupby(group_cols).apply(create_rolling_for_group).reset_index(drop=True)
        else:
            df = create_rolling_for_group(df)
        
        return df
    
    def create_seasonal_features(self, df: pd.DataFrame,
                               timestamp_col: str) -> pd.DataFrame:
        """
        Create seasonal and calendar features.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            pd.DataFrame: DataFrame with seasonal features added
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if df[timestamp_col].dtype == 'object':
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Ensure we have basic time features
        if 'month' not in df.columns:
            df = self.create_basic_time_features(df, timestamp_col)
        
        # Weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Saturday=5, Sunday=6
        
        # Holiday indicator
        df['is_holiday'] = df[timestamp_col].dt.date.isin(self.holiday_calendar).astype(int)
        
        # Season (meteorological seasons)
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df['season'] = df['month'].apply(get_season)
        
        # Time of day categories
        def get_time_of_day(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        
        df['time_of_day'] = df['hour'].apply(get_time_of_day)
        
        # Working hours indicator
        df['is_working_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17) & ~df['is_weekend']).astype(int)
        
        return df
    
    def create_temporal_aggregations(self, df: pd.DataFrame,
                                   target_col: str,
                                   timestamp_col: str,
                                   group_cols: List[str] = None,
                                   time_groupings: List[str] = None) -> pd.DataFrame:
        """
        Create temporal aggregation features.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            timestamp_col: Name of timestamp column
            group_cols: Columns to group by (e.g., vessel_id)
            time_groupings: List of time groupings ('hour', 'day', 'week', 'month')
            
        Returns:
            pd.DataFrame: DataFrame with temporal aggregation features added
        """
        if time_groupings is None:
            time_groupings = ['hour', 'day_of_week']
        
        if group_cols is None:
            group_cols = []
        
        df = df.copy()
        
        # Ensure we have basic time features
        if 'hour' not in df.columns:
            df = self.create_basic_time_features(df, timestamp_col)
        
        for time_group in time_groupings:
            if time_group in df.columns:
                # Calculate aggregations by time grouping
                agg_group_cols = group_cols + [time_group]
                
                # Mean by time group
                mean_agg = df.groupby(agg_group_cols)[target_col].mean().reset_index()
                mean_agg = mean_agg.rename(columns={target_col: f'{target_col}_mean_by_{time_group}'})
                df = df.merge(mean_agg, on=agg_group_cols, how='left')
                
                # Standard deviation by time group
                std_agg = df.groupby(agg_group_cols)[target_col].std().reset_index()
                std_agg = std_agg.rename(columns={target_col: f'{target_col}_std_by_{time_group}'})
                df = df.merge(std_agg, on=agg_group_cols, how='left')
                
                # Count by time group
                count_agg = df.groupby(agg_group_cols)[target_col].count().reset_index()
                count_agg = count_agg.rename(columns={target_col: f'{target_col}_count_by_{time_group}'})
                df = df.merge(count_agg, on=agg_group_cols, how='left')
        
        return df
    
    def create_time_since_features(self, df: pd.DataFrame,
                                 timestamp_col: str,
                                 event_cols: List[str] = None,
                                 group_cols: List[str] = None) -> pd.DataFrame:
        """
        Create time-since-event features.
        
        Args:
            df: DataFrame with time series data
            timestamp_col: Name of timestamp column
            event_cols: List of event columns (binary indicators)
            group_cols: Columns to group by (e.g., vessel_id)
            
        Returns:
            pd.DataFrame: DataFrame with time-since features added
        """
        if event_cols is None:
            event_cols = ['is_weekend', 'is_holiday']
        
        if group_cols is None:
            group_cols = []
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        if df[timestamp_col].dtype == 'object':
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by group columns and timestamp
        sort_cols = group_cols + [timestamp_col]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        
        def create_time_since_for_group(group_df):
            group_df = group_df.copy()
            
            for event_col in event_cols:
                if event_col in group_df.columns:
                    # Initialize time since last event
                    time_since_col = f'time_since_{event_col}'
                    group_df[time_since_col] = np.nan
                    
                    last_event_time = None
                    
                    for idx, row in group_df.iterrows():
                        current_time = row[timestamp_col]
                        
                        if row[event_col] == 1:
                            last_event_time = current_time
                            group_df.loc[idx, time_since_col] = 0
                        elif last_event_time is not None:
                            time_diff = (current_time - last_event_time).total_seconds() / 3600  # hours
                            group_df.loc[idx, time_since_col] = time_diff
            
            return group_df
        
        if group_cols:
            df = df.groupby(group_cols).apply(create_time_since_for_group).reset_index(drop=True)
        else:
            df = create_time_since_for_group(df)
        
        return df
