"""
Geospatial feature engineering for AIS data.
"""

import pandas as pd
import numpy as np
import h3
from typing import List, Dict, Any, Optional
from geopy.distance import geodesic
import logging


class GeoFeatureEngineer:
    """
    Class for creating geospatial features from AIS data.
    """
    
    def __init__(self, h3_resolution: int = 8):
        """
        Initialize the geospatial feature engineer.
        
        Args:
            h3_resolution: H3 grid resolution (0-15, higher = more granular)
        """
        self.h3_resolution = h3_resolution
        self.logger = logging.getLogger(__name__)
    
    def create_h3_cells(self, df: pd.DataFrame, 
                       lat_col: str = 'lat', 
                       lon_col: str = 'lon') -> pd.DataFrame:
        """
        Create H3 cell identifiers from latitude and longitude.
        
        Args:
            df: DataFrame with latitude and longitude columns
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            
        Returns:
            pd.DataFrame: DataFrame with H3 cell column added
        """
        df = df.copy()
        
        # Validate coordinates
        invalid_coords = (
            (df[lat_col] < -90) | (df[lat_col] > 90) |
            (df[lon_col] < -180) | (df[lon_col] > 180) |
            df[lat_col].isna() | df[lon_col].isna()
        )
        
        if invalid_coords.any():
            self.logger.warning(f"Found {invalid_coords.sum()} invalid coordinates, setting H3 cells to NaN")
        
        # Create H3 cells
        def get_h3_cell(lat, lon):
            try:
                if pd.isna(lat) or pd.isna(lon):
                    return None
                if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                    return None
                return h3.geo_to_h3(lat, lon, self.h3_resolution)
            except Exception:
                return None
        
        df['h3_cell'] = df.apply(lambda row: get_h3_cell(row[lat_col], row[lon_col]), axis=1)
        
        return df
    
    def create_h3_center_coordinates(self, df: pd.DataFrame, 
                                   h3_col: str = 'h3_cell') -> pd.DataFrame:
        """
        Create center coordinates for H3 cells.
        
        Args:
            df: DataFrame with H3 cell column
            h3_col: Name of H3 cell column
            
        Returns:
            pd.DataFrame: DataFrame with H3 center coordinates added
        """
        df = df.copy()
        
        def get_h3_center(h3_cell):
            try:
                if pd.isna(h3_cell) or h3_cell is None:
                    return None, None
                lat, lon = h3.h3_to_geo(h3_cell)
                return lat, lon
            except Exception:
                return None, None
        
        # Get center coordinates
        centers = df[h3_col].apply(get_h3_center)
        df['h3_center_lat'] = [center[0] for center in centers]
        df['h3_center_lon'] = [center[1] for center in centers]
        
        return df
    
    def create_h3_neighbors(self, df: pd.DataFrame, 
                          h3_col: str = 'h3_cell',
                          k_ring: int = 1) -> pd.DataFrame:
        """
        Create H3 neighbor information.
        
        Args:
            df: DataFrame with H3 cell column
            h3_col: Name of H3 cell column
            k_ring: Ring distance for neighbors (1 = immediate neighbors)
            
        Returns:
            pd.DataFrame: DataFrame with neighbor information added
        """
        df = df.copy()
        
        def get_neighbors(h3_cell):
            try:
                if pd.isna(h3_cell) or h3_cell is None:
                    return []
                return list(h3.k_ring(h3_cell, k_ring))
            except Exception:
                return []
        
        df[f'h3_neighbors_k{k_ring}'] = df[h3_col].apply(get_neighbors)
        df[f'h3_neighbor_count_k{k_ring}'] = df[f'h3_neighbors_k{k_ring}'].apply(len)
        
        return df
    
    def calculate_distance_features(self, df: pd.DataFrame,
                                  lat_col: str = 'lat',
                                  lon_col: str = 'lon',
                                  group_cols: List[str] = None) -> pd.DataFrame:
        """
        Calculate distance-based features.
        
        Args:
            df: DataFrame with position data
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            group_cols: Columns to group by (e.g., vessel_id)
            
        Returns:
            pd.DataFrame: DataFrame with distance features added
        """
        df = df.copy()
        
        if group_cols is None:
            group_cols = []
        
        # Sort by group columns and timestamp if available
        sort_cols = group_cols.copy()
        if 'timestamp' in df.columns:
            sort_cols.append('timestamp')
        elif 'time_idx' in df.columns:
            sort_cols.append('time_idx')
        
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        
        # Calculate distance from previous point
        def calculate_distance_from_previous(group_df):
            distances = [np.nan]  # First point has no previous point
            
            for i in range(1, len(group_df)):
                try:
                    prev_coords = (group_df.iloc[i-1][lat_col], group_df.iloc[i-1][lon_col])
                    curr_coords = (group_df.iloc[i][lat_col], group_df.iloc[i][lon_col])
                    
                    if any(pd.isna(coord) for coord in prev_coords + curr_coords):
                        distances.append(np.nan)
                    else:
                        distance = geodesic(prev_coords, curr_coords).meters
                        distances.append(distance)
                except Exception:
                    distances.append(np.nan)
            
            group_df = group_df.copy()
            group_df['distance_from_previous'] = distances
            return group_df
        
        if group_cols:
            df = df.groupby(group_cols).apply(calculate_distance_from_previous).reset_index(drop=True)
        else:
            df = calculate_distance_from_previous(df)
        
        return df
    
    def create_speed_features(self, df: pd.DataFrame,
                            lat_col: str = 'lat',
                            lon_col: str = 'lon',
                            timestamp_col: str = 'timestamp',
                            group_cols: List[str] = None) -> pd.DataFrame:
        """
        Calculate speed features from position and time data.
        
        Args:
            df: DataFrame with position and time data
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            timestamp_col: Name of timestamp column
            group_cols: Columns to group by (e.g., vessel_id)
            
        Returns:
            pd.DataFrame: DataFrame with speed features added
        """
        df = df.copy()
        
        if group_cols is None:
            group_cols = []
        
        # Ensure timestamp is datetime
        if df[timestamp_col].dtype == 'object':
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by group columns and timestamp
        sort_cols = group_cols + [timestamp_col]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        
        def calculate_speed(group_df):
            speeds = [np.nan]  # First point has no speed
            
            for i in range(1, len(group_df)):
                try:
                    prev_coords = (group_df.iloc[i-1][lat_col], group_df.iloc[i-1][lon_col])
                    curr_coords = (group_df.iloc[i][lat_col], group_df.iloc[i][lon_col])
                    prev_time = group_df.iloc[i-1][timestamp_col]
                    curr_time = group_df.iloc[i][timestamp_col]
                    
                    if (any(pd.isna(coord) for coord in prev_coords + curr_coords) or
                        pd.isna(prev_time) or pd.isna(curr_time)):
                        speeds.append(np.nan)
                        continue
                    
                    # Calculate distance in meters
                    distance = geodesic(prev_coords, curr_coords).meters
                    
                    # Calculate time difference in seconds
                    time_diff = (curr_time - prev_time).total_seconds()
                    
                    if time_diff > 0:
                        # Speed in m/s
                        speed = distance / time_diff
                        speeds.append(speed)
                    else:
                        speeds.append(np.nan)
                        
                except Exception:
                    speeds.append(np.nan)
            
            group_df = group_df.copy()
            group_df['calculated_speed'] = speeds
            
            # Convert to knots (1 m/s = 1.94384 knots)
            group_df['calculated_speed_knots'] = group_df['calculated_speed'] * 1.94384
            
            return group_df
        
        if group_cols:
            df = df.groupby(group_cols).apply(calculate_speed).reset_index(drop=True)
        else:
            df = calculate_speed(df)
        
        return df
    
    def create_bearing_features(self, df: pd.DataFrame,
                              lat_col: str = 'lat',
                              lon_col: str = 'lon',
                              group_cols: List[str] = None) -> pd.DataFrame:
        """
        Calculate bearing (course) features.
        
        Args:
            df: DataFrame with position data
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            group_cols: Columns to group by (e.g., vessel_id)
            
        Returns:
            pd.DataFrame: DataFrame with bearing features added
        """
        df = df.copy()
        
        if group_cols is None:
            group_cols = []
        
        # Sort by group columns and timestamp if available
        sort_cols = group_cols.copy()
        if 'timestamp' in df.columns:
            sort_cols.append('timestamp')
        elif 'time_idx' in df.columns:
            sort_cols.append('time_idx')
        
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        
        def calculate_bearing(lat1, lon1, lat2, lon2):
            """Calculate bearing between two points."""
            try:
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                
                dlon = lon2 - lon1
                
                y = np.sin(dlon) * np.cos(lat2)
                x = (np.cos(lat1) * np.sin(lat2) - 
                     np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
                
                bearing = np.degrees(np.arctan2(y, x))
                
                # Normalize to 0-360 degrees
                bearing = (bearing + 360) % 360
                
                return bearing
            except Exception:
                return np.nan
        
        def calculate_bearing_features(group_df):
            bearings = [np.nan]  # First point has no bearing
            
            for i in range(1, len(group_df)):
                prev_lat = group_df.iloc[i-1][lat_col]
                prev_lon = group_df.iloc[i-1][lon_col]
                curr_lat = group_df.iloc[i][lat_col]
                curr_lon = group_df.iloc[i][lon_col]
                
                bearing = calculate_bearing(prev_lat, prev_lon, curr_lat, curr_lon)
                bearings.append(bearing)
            
            group_df = group_df.copy()
            group_df['bearing'] = bearings
            
            # Create cyclical features for bearing
            group_df['bearing_sin'] = np.sin(np.radians(group_df['bearing']))
            group_df['bearing_cos'] = np.cos(np.radians(group_df['bearing']))
            
            return group_df
        
        if group_cols:
            df = df.groupby(group_cols).apply(calculate_bearing_features).reset_index(drop=True)
        else:
            df = calculate_bearing_features(df)
        
        return df
    
    def create_spatial_aggregations(self, df: pd.DataFrame,
                                  h3_col: str = 'h3_cell',
                                  value_cols: List[str] = None,
                                  time_col: str = 'timestamp',
                                  agg_functions: List[str] = None) -> pd.DataFrame:
        """
        Create spatial aggregation features based on H3 cells.
        
        Args:
            df: DataFrame with H3 cell data
            h3_col: Name of H3 cell column
            value_cols: Columns to aggregate
            time_col: Time column for temporal grouping
            agg_functions: Aggregation functions to apply
            
        Returns:
            pd.DataFrame: DataFrame with spatial aggregation features added
        """
        if value_cols is None:
            value_cols = ['calculated_speed']
        
        if agg_functions is None:
            agg_functions = ['mean', 'std', 'count']
        
        df = df.copy()
        
        # Create aggregations by H3 cell
        agg_dict = {}
        for col in value_cols:
            if col in df.columns:
                for func in agg_functions:
                    agg_dict[f'{col}_{func}_by_h3'] = (col, func)
        
        if agg_dict:
            h3_aggs = df.groupby(h3_col).agg(agg_dict).reset_index()
            h3_aggs.columns = [h3_col] + [f'{col}_{func}_by_h3' for col, func in agg_dict.values()]
            
            # Merge back to original dataframe
            df = df.merge(h3_aggs, on=h3_col, how='left')
        
        return df
