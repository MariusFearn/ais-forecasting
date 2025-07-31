"""
Vessel Feature Engineering - Individual vessel features for H3 sequence modeling

This module implements Phase 2 of the vessel-level H3 feature engineering,
creating comprehensive features for individual vessel behavior analysis and prediction.
"""

import pandas as pd
import numpy as np
import h3
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from geopy.distance import geodesic
import math


class VesselFeatureExtractor:
    """
    Extract comprehensive features for individual vessels in H3 space
    
    This class implements Phase 2.1-2.3 features:
    - Core vessel journey features
    - Movement pattern features  
    - Contextual features
    """
    
    def __init__(self, h3_resolution: int = 5):
        """
        Initialize the vessel feature extractor
        
        Args:
            h3_resolution: H3 resolution level for spatial features
        """
        self.h3_resolution = h3_resolution
        self.edge_length_km = h3.edge_length(h3_resolution, unit='km')
        
    def extract_all_features(self, vessel_h3_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all vessel features from H3 sequence data
        
        Args:
            vessel_h3_data: DataFrame with vessel H3 sequence (from VesselH3Tracker)
            
        Returns:
            DataFrame with comprehensive vessel features
        """
        df = vessel_h3_data.copy().sort_values('mdt').reset_index(drop=True)
        
        # Core journey features
        df = self._add_current_state_features(df)
        df = self._add_historical_sequence_features(df)
        
        # Movement pattern features
        df = self._add_direction_speed_patterns(df)
        df = self._add_journey_characteristics(df)
        
        # Contextual features
        df = self._add_geographic_context(df)
        df = self._add_operational_context(df)
        
        return df
    
    def _add_current_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add current state features (Phase 2.1)
        """
        # Current position and movement
        df['current_h3_cell'] = df['h3_cell']
        df['current_speed'] = df['speed'] if 'speed' in df.columns else np.nan
        df['current_heading'] = df['heading'] if 'heading' in df.columns else np.nan
        
        # Time in current cell
        df['cell_group'] = (df['h3_cell'] != df['h3_cell'].shift()).cumsum()
        df['time_in_current_cell'] = df.groupby('cell_group').cumcount() + 1
        
        # Time since cell entry
        if 'time_diff_hours' in df.columns:
            df['time_in_cell_hours'] = df.groupby('cell_group')['time_diff_hours'].cumsum()
        
        return df
    
    def _add_historical_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add historical sequence features (Phase 2.1)
        """
        # H3 sequence features - count unique cells in window using a manual approach
        for window in [6, 12, 24]:
            # Manual rolling unique count to avoid pandas issues
            unique_counts = []
            for i in range(len(df)):
                start_idx = max(0, i - window + 1)
                window_cells = df.iloc[start_idx:i+1]['h3_cell'].dropna()
                unique_counts.append(len(set(window_cells)))
            df[f'cells_visited_{window}h'] = unique_counts
        
        # Speed sequence features
        if 'speed' in df.columns and pd.api.types.is_numeric_dtype(df['speed']):
            for window in [6, 12, 24]:
                df[f'avg_speed_{window}h'] = df['speed'].rolling(
                    window=window, min_periods=1
                ).mean()
        
        # Add cell transition rate using simple rolling sum
        if 'h3_cell_changed' in df.columns:
            df['cell_transitions_6h'] = df['h3_cell_changed'].rolling(
                window=6, min_periods=1
            ).sum()
        
        return df
    
    def _add_direction_speed_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add direction and speed pattern features (Phase 2.2)
        """
        if 'speed' not in df.columns or not pd.api.types.is_numeric_dtype(df['speed']):
            return df
            
        # Speed trends - simplified calculation
        for window in [6, 12]:
            df[f'speed_trend_{window}h'] = df['speed'].rolling(
                window=window, min_periods=2
            ).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else 0, raw=False)
        
        # Speed variability
        for window in [6, 12]:
            df[f'speed_std_{window}h'] = df['speed'].rolling(
                window=window, min_periods=1
            ).std()
        
        # Heading consistency (if available)
        if 'heading' in df.columns and pd.api.types.is_numeric_dtype(df['heading']):
            for window in [6, 12]:
                df[f'heading_consistency_{window}h'] = df['heading'].rolling(
                    window=window, min_periods=1
                ).std()
        
        return df
    
    def _add_journey_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add journey characteristic features (Phase 2.2)
        """
        # Cumulative journey metrics
        df['total_journey_time'] = (df['mdt'] - df['mdt'].iloc[0]).dt.total_seconds() / 3600
        
        # Simplified cells visited cumulative using manual approach
        unique_cumulative = []
        seen_cells = set()
        for cell in df['h3_cell']:
            if pd.notna(cell):
                seen_cells.add(cell)
            unique_cumulative.append(len(seen_cells))
        df['cells_visited_cumulative'] = unique_cumulative
        
        # Distance from journey start
        if len(df) > 0 and 'lat' in df.columns and 'lon' in df.columns:
            start_lat, start_lon = df.iloc[0]['lat'], df.iloc[0]['lon']
            df['distance_from_start_km'] = df.apply(
                lambda row: geodesic((start_lat, start_lon), (row['lat'], row['lon'])).kilometers
                if pd.notna(row['lat']) and pd.notna(row['lon']) else 0,
                axis=1
            )
        else:
            df['distance_from_start_km'] = 0
        
        # Journey phases (basic classification)
        df['journey_phase'] = self._classify_journey_phase(df)
        
        # Port departure/approach detection
        df['likely_port_departure'] = self._detect_port_events(df, event_type='departure')
        df['likely_port_approach'] = self._detect_port_events(df, event_type='approach')
        
        return df
    
    def _add_geographic_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add geographic context features (Phase 2.3)
        
        Note: Advanced geographic features requiring external data sources
        have been moved to future implementation phase.
        """
        # Regional classification - this works well with lat/lon
        df['ocean_region'] = df.apply(
            lambda row: self._classify_ocean_region(row['lat'], row['lon']), axis=1
        )
        
        return df
    
    def _add_operational_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add operational context features (Phase 2.3)
        
        Note: Complex operational features requiring domain knowledge
        have been moved to future implementation phase.
        """        
        # Time-based operational features - these are reliable
        df['hour_of_day'] = df['mdt'].dt.hour
        df['day_of_week'] = df['mdt'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    # Helper methods for feature calculation
    
    def _calculate_movement_efficiency(self, window_data) -> float:
        """Calculate movement efficiency over a time window"""
        if len(window_data) < 2:
            return 1.0
            
        try:
            # Get first and last positions
            start_pos = (window_data.iloc[0]['lat'], window_data.iloc[0]['lon'])
            end_pos = (window_data.iloc[-1]['lat'], window_data.iloc[-1]['lon'])
            
            # Direct distance
            direct_distance = geodesic(start_pos, end_pos).kilometers
            
            # Actual path distance (sum of segments)
            actual_distance = 0
            for i in range(1, len(window_data)):
                pos1 = (window_data.iloc[i-1]['lat'], window_data.iloc[i-1]['lon'])
                pos2 = (window_data.iloc[i]['lat'], window_data.iloc[i]['lon'])
                actual_distance += geodesic(pos1, pos2).kilometers
            
            # Efficiency ratio
            if actual_distance > 0:
                return direct_distance / actual_distance
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _classify_journey_phase(self, df: pd.DataFrame) -> pd.Series:
        """Classify journey phases based on speed and movement patterns"""
        phases = []
        
        for i, row in df.iterrows():
            speed = row.get('speed', 0)
            
            if speed < 1:
                phase = 'stationary'
            elif speed < 5:
                phase = 'slow_maneuvering'
            elif speed < 12:
                phase = 'transit_slow'
            else:
                phase = 'transit_fast'
                
            phases.append(phase)
        
        return pd.Series(phases, index=df.index)
    
    def _detect_port_events(self, df: pd.DataFrame, event_type: str) -> pd.Series:
        """Detect port departure/approach events"""
        # Simplified detection based on speed patterns
        events = []
        
        for i, row in df.iterrows():
            if i < 3:
                events.append(0)
                continue
                
            # Look at speed pattern in last few hours
            recent_speeds = df.iloc[max(0, i-3):i+1]['speed'].values if 'speed' in df.columns else [0]
            
            if event_type == 'departure':
                # Speed increasing from low to higher
                if len(recent_speeds) >= 3 and recent_speeds[0] < 2 and recent_speeds[-1] > 8:
                    events.append(1)
                else:
                    events.append(0)
            else:  # approach
                # Speed decreasing from higher to low
                if len(recent_speeds) >= 3 and recent_speeds[0] > 8 and recent_speeds[-1] < 2:
                    events.append(1)
                else:
                    events.append(0)
        
        return pd.Series(events, index=df.index)
    
    def _estimate_coastal_distance(self, lat: float, lon: float) -> float:
        """Rough estimation of distance from coast (simplified)"""
        # Very simplified - just use distance from known coastal reference points
        # In practice, you'd use actual coastline data
        
        # Cape Town area reference
        cape_town = (-33.9249, 18.4241)
        distance = geodesic((lat, lon), cape_town).kilometers
        
        # Rough estimation: closer to reference = closer to coast
        return min(distance, 500)  # Cap at 500km
    
    def _estimate_water_depth(self, lat: float, lon: float) -> float:
        """Very rough water depth estimation"""
        # Simplified: use distance from coast as proxy
        coastal_distance = self._estimate_coastal_distance(lat, lon)
        
        # Rough approximation: depth increases with distance from coast
        if coastal_distance < 10:
            return 20  # Shallow coastal waters
        elif coastal_distance < 50:
            return 100  # Continental shelf
        else:
            return 2000  # Deep ocean
    
    def _classify_ocean_region(self, lat: float, lon: float) -> str:
        """Classify ocean region"""
        # Cape Town area classification
        if -35 < lat < -30 and 15 < lon < 25:
            return 'cape_town_approaches'
        elif -40 < lat < -30:
            return 'southern_atlantic'
        elif -30 < lat < -20:
            return 'south_atlantic'
        else:
            return 'other'
    
    def _detect_shipping_lanes(self, df: pd.DataFrame) -> pd.Series:
        """Detect if vessel is in major shipping lanes"""
        # Simplified: high traffic areas based on vessel density
        # In practice, use actual shipping lane data
        
        lane_indicators = []
        for i, row in df.iterrows():
            # Simple heuristic: consistent heading and good speed
            if i >= 6:
                recent_speeds = df.iloc[i-5:i+1]['speed'].values if 'speed' in df.columns else [0]
                avg_speed = np.mean(recent_speeds)
                
                if 8 < avg_speed < 18:  # Typical shipping lane speeds
                    lane_indicators.append(1)
                else:
                    lane_indicators.append(0)
            else:
                lane_indicators.append(0)
        
        return pd.Series(lane_indicators, index=df.index)
    
    def _estimate_cargo_status(self, df: pd.DataFrame) -> pd.Series:
        """Estimate cargo loading status"""
        statuses = []
        
        for i, row in df.iterrows():
            speed = row.get('speed', 0)
            draught = row.get('draught', 10)  # Default draught
            
            # Simple heuristics
            if speed < 1:
                if draught > 12:
                    status = 'loaded_stationary'
                else:
                    status = 'ballast_stationary'
            elif speed > 12:
                if draught > 12:
                    status = 'loaded_transit'
                else:
                    status = 'ballast_transit'
            else:
                status = 'maneuvering'
            
            statuses.append(status)
        
        return pd.Series(statuses, index=df.index)
    
    def _analyze_port_approach_behavior(self, df: pd.DataFrame) -> pd.Series:
        """Analyze speed patterns during port approaches"""
        behaviors = []
        
        for i, row in df.iterrows():
            if i < 6:
                behaviors.append('insufficient_data')
                continue
            
            # Look at speed pattern in last 6 hours
            recent_speeds = df.iloc[i-5:i+1]['speed'].values if 'speed' in df.columns else [10]
            
            if len(recent_speeds) >= 6:
                # Analyze speed pattern
                if np.all(recent_speeds < 3):
                    behavior = 'stationary_pattern'
                elif recent_speeds[0] > 10 and recent_speeds[-1] < 5:
                    behavior = 'deceleration_pattern'
                elif recent_speeds[0] < 5 and recent_speeds[-1] > 10:
                    behavior = 'acceleration_pattern'
                else:
                    behavior = 'steady_pattern'
            else:
                behavior = 'insufficient_data'
            
            behaviors.append(behavior)
        
        return pd.Series(behaviors, index=df.index)
    
    def _detect_anchorage_periods(self, df: pd.DataFrame) -> pd.Series:
        """Detect time spent in anchorage (stationary periods)"""
        anchorage_times = []
        
        # Group by stationary periods
        df['is_stationary'] = (df['speed'] < 1) if 'speed' in df.columns else False
        df['stationary_group'] = (df['is_stationary'] != df['is_stationary'].shift()).cumsum()
        
        for i, row in df.iterrows():
            if row['is_stationary']:
                # Calculate time in current stationary period
                group = row['stationary_group']
                group_data = df[df['stationary_group'] == group]
                if len(group_data) > 1:
                    duration = (group_data['mdt'].max() - group_data['mdt'].min()).total_seconds() / 3600
                    anchorage_times.append(duration)
                else:
                    anchorage_times.append(0)
            else:
                anchorage_times.append(0)
        
        return pd.Series(anchorage_times, index=df.index)


def create_lag_features(df: pd.DataFrame, feature_cols: List[str], 
                       lag_periods: List[int] = [1, 6, 12, 24]) -> pd.DataFrame:
    """
    Create lag features for specified columns
    
    Args:
        df: DataFrame with features
        feature_cols: List of column names to create lags for
        lag_periods: List of lag periods (in hours/records)
        
    Returns:
        DataFrame with additional lag features
    """
    df_with_lags = df.copy()
    
    for col in feature_cols:
        if col in df.columns:
            for lag in lag_periods:
                lag_col_name = f"{col}_lag_{lag}h"
                df_with_lags[lag_col_name] = df[col].shift(lag)
    
    return df_with_lags


def create_rolling_features(df: pd.DataFrame, feature_cols: List[str],
                          windows: List[int] = [6, 12, 24]) -> pd.DataFrame:
    """
    Create rolling statistical features
    
    Args:
        df: DataFrame with features
        feature_cols: List of column names to create rolling features for
        windows: List of window sizes (in hours/records)
        
    Returns:
        DataFrame with additional rolling features
    """
    df_with_rolling = df.copy()
    
    for col in feature_cols:
        if col in df.columns:
            for window in windows:
                # Rolling mean
                df_with_rolling[f"{col}_rolling_mean_{window}h"] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                df_with_rolling[f"{col}_rolling_std_{window}h"] = df[col].rolling(
                    window=window, min_periods=1
                ).std()
                
                # Rolling min/max
                df_with_rolling[f"{col}_rolling_min_{window}h"] = df[col].rolling(
                    window=window, min_periods=1
                ).min()
                
                df_with_rolling[f"{col}_rolling_max_{window}h"] = df[col].rolling(
                    window=window, min_periods=1
                ).max()
    
    return df_with_rolling


if __name__ == "__main__":
    print("VesselFeatureExtractor module loaded successfully")
    print("Usage:")
    print("  extractor = VesselFeatureExtractor(h3_resolution=5)")
    print("  features_df = extractor.extract_all_features(vessel_h3_data)")
