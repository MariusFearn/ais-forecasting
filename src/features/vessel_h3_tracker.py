"""
Vessel H3 Tracker - Core vessel tracking logic for H3 feature engineering

This module implements the foundation for vessel-level H3 feature engineering,
focusing on individual capsize vessel movement prediction through H3 grid cells.
"""

import pandas as pd
import numpy as np
import h3
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings


class VesselH3Tracker:
    """
    Core class for tracking individual vessels through H3 grid cells
    
    This class handles the conversion of vessel AIS data into H3 sequences
    and provides basic journey analysis capabilities.
    """
    
    def __init__(self, h3_resolution: int = 5):
        """
        Initialize the vessel H3 tracker
        
        Args:
            h3_resolution: H3 resolution level (default: 5, ~8.5km edge length)
        """
        self.h3_resolution = h3_resolution
        self.edge_length_km = h3.edge_length(h3_resolution, unit='km')
        
        print(f"VesselH3Tracker initialized with resolution {h3_resolution}")
        print(f"Average H3 cell edge length: {self.edge_length_km:.2f} km")
    
    def convert_vessel_to_h3_sequence(self, vessel_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert vessel AIS data to H3 sequence with basic features
        
        Args:
            vessel_data: DataFrame with vessel AIS data (must have lat, lon, mdt columns)
        
        Returns:
            DataFrame with H3 cells and basic sequence features
        """
        # Validate input data
        required_cols = ['lat', 'lon', 'mdt']
        missing_cols = [col for col in required_cols if col not in vessel_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by time and copy data
        df = vessel_data.copy().sort_values('mdt').reset_index(drop=True)
        
        # Convert coordinates to H3 cells
        h3_cells = []
        conversion_errors = 0
        
        for _, row in df.iterrows():
            try:
                if pd.notna(row['lat']) and pd.notna(row['lon']):
                    # Validate coordinate ranges
                    if abs(row['lat']) <= 90 and abs(row['lon']) <= 180:
                        cell = h3.geo_to_h3(row['lat'], row['lon'], self.h3_resolution)
                        h3_cells.append(cell)
                    else:
                        h3_cells.append(None)
                        conversion_errors += 1
                else:
                    h3_cells.append(None)
                    conversion_errors += 1
            except Exception as e:
                h3_cells.append(None)
                conversion_errors += 1
        
        if conversion_errors > 0:
            warnings.warn(f"Failed to convert {conversion_errors} coordinates to H3 cells")
        
        df['h3_cell'] = h3_cells
        
        # Add basic sequence features
        df = self._add_sequence_features(df)
        
        return df
    
    def _add_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic sequence features to H3 data
        
        Args:
            df: DataFrame with h3_cell column
            
        Returns:
            DataFrame with additional sequence features
        """
        # Previous cell and transition indicators
        df['h3_cell_prev'] = df['h3_cell'].shift(1)
        df['h3_cell_changed'] = (df['h3_cell'] != df['h3_cell_prev']) & df['h3_cell'].notna()
        
        # Time-based features
        df['time_diff_seconds'] = df['mdt'].diff().dt.total_seconds()
        df['time_diff_hours'] = df['time_diff_seconds'] / 3600
        
        # Cell transition features
        df['cell_transition'] = df['h3_cell_changed'].astype(int)
        df['cells_visited_cumulative'] = df['h3_cell_changed'].cumsum()
        
        # Time in current cell (group by consecutive cell periods)
        df['cell_group'] = (df['h3_cell'] != df['h3_cell'].shift()).cumsum()
        df['time_in_cell_cumulative'] = df.groupby('cell_group')['time_diff_hours'].cumsum()
        
        return df
    
    def analyze_vessel_journey(self, vessel_h3_data: pd.DataFrame) -> Dict:
        """
        Analyze vessel journey patterns in H3 space
        
        Args:
            vessel_h3_data: DataFrame with H3 sequence data
            
        Returns:
            Dictionary with journey analysis results
        """
        if vessel_h3_data.empty:
            return {'error': 'Empty vessel data provided'}
        
        # Basic journey stats
        analysis = {
            'vessel_id': vessel_h3_data['imo'].iloc[0] if 'imo' in vessel_h3_data.columns else 'Unknown',
            'total_records': len(vessel_h3_data),
            'start_time': vessel_h3_data['mdt'].min(),
            'end_time': vessel_h3_data['mdt'].max(),
            'journey_duration_days': (vessel_h3_data['mdt'].max() - vessel_h3_data['mdt'].min()).days,
            'unique_h3_cells': vessel_h3_data['h3_cell'].nunique(),
            'total_cell_transitions': vessel_h3_data['cell_transition'].sum(),
            'records_per_cell': len(vessel_h3_data) / vessel_h3_data['h3_cell'].nunique() if vessel_h3_data['h3_cell'].nunique() > 0 else 0
        }
        
        # Geographic coverage
        analysis.update({
            'lat_min': vessel_h3_data['lat'].min(),
            'lat_max': vessel_h3_data['lat'].max(),
            'lon_min': vessel_h3_data['lon'].min(),
            'lon_max': vessel_h3_data['lon'].max(),
            'lat_range_degrees': vessel_h3_data['lat'].max() - vessel_h3_data['lat'].min(),
            'lon_range_degrees': vessel_h3_data['lon'].max() - vessel_h3_data['lon'].min()
        })
        
        # Movement patterns
        if 'speed' in vessel_h3_data.columns:
            analysis.update({
                'avg_speed_knots': vessel_h3_data['speed'].mean(),
                'max_speed_knots': vessel_h3_data['speed'].max(),
                'stationary_time_pct': (vessel_h3_data['speed'] < 1).mean() * 100,
                'high_speed_time_pct': (vessel_h3_data['speed'] > 15).mean() * 100
            })
        
        # Cell timing analysis
        if 'time_in_cell_cumulative' in vessel_h3_data.columns:
            cell_times = vessel_h3_data.groupby('h3_cell')['time_diff_hours'].sum()
            analysis.update({
                'avg_time_per_cell_hours': cell_times.mean(),
                'max_time_per_cell_hours': cell_times.max(),
                'min_time_per_cell_hours': cell_times.min()
            })
        
        # Data quality metrics
        analysis.update({
            'missing_h3_cells': vessel_h3_data['h3_cell'].isna().sum(),
            'missing_h3_pct': vessel_h3_data['h3_cell'].isna().mean() * 100,
            'median_time_gap_hours': vessel_h3_data['time_diff_hours'].median()
        })
        
        return analysis
    
    def get_vessel_h3_sequence(self, vessel_h3_data: pd.DataFrame, 
                              include_timestamps: bool = True) -> List[str]:
        """
        Extract clean H3 cell sequence for a vessel
        
        Args:
            vessel_h3_data: DataFrame with H3 sequence data
            include_timestamps: Whether to include timestamp information
            
        Returns:
            List of H3 cell IDs in chronological order
        """
        # Filter out missing cells and sort by time
        valid_data = vessel_h3_data[vessel_h3_data['h3_cell'].notna()].sort_values('mdt')
        
        if include_timestamps:
            return list(zip(valid_data['h3_cell'].tolist(), 
                          valid_data['mdt'].tolist()))
        else:
            return valid_data['h3_cell'].tolist()
    
    def identify_stationary_periods(self, vessel_h3_data: pd.DataFrame, 
                                  min_duration_hours: float = 6.0) -> pd.DataFrame:
        """
        Identify periods where vessel remained in same H3 cell for extended time
        
        Args:
            vessel_h3_data: DataFrame with H3 sequence data
            min_duration_hours: Minimum duration to consider as stationary period
            
        Returns:
            DataFrame with stationary periods
        """
        if 'speed' not in vessel_h3_data.columns:
            warnings.warn("Speed data not available, using H3 cell changes only")
        
        # Group by consecutive periods in same cell
        vessel_h3_data['cell_group'] = (vessel_h3_data['h3_cell'] != vessel_h3_data['h3_cell'].shift()).cumsum()
        
        stationary_periods = []
        
        for group_id, group_data in vessel_h3_data.groupby('cell_group'):
            if len(group_data) < 2 or group_data['h3_cell'].iloc[0] is None:
                continue
                
            duration_hours = (group_data['mdt'].max() - group_data['mdt'].min()).total_seconds() / 3600
            
            if duration_hours >= min_duration_hours:
                # Additional check: low speed if available
                low_speed = True
                if 'speed' in group_data.columns:
                    avg_speed = group_data['speed'].mean()
                    low_speed = avg_speed < 3.0  # knots
                
                if low_speed:
                    stationary_periods.append({
                        'h3_cell': group_data['h3_cell'].iloc[0],
                        'start_time': group_data['mdt'].min(),
                        'end_time': group_data['mdt'].max(),
                        'duration_hours': duration_hours,
                        'avg_speed': group_data['speed'].mean() if 'speed' in group_data.columns else None,
                        'record_count': len(group_data),
                        'center_lat': group_data['lat'].mean(),
                        'center_lon': group_data['lon'].mean()
                    })
        
        return pd.DataFrame(stationary_periods)
    
    def validate_h3_sequence_quality(self, vessel_h3_data: pd.DataFrame) -> Dict:
        """
        Validate the quality of H3 sequence data
        
        Args:
            vessel_h3_data: DataFrame with H3 sequence data
            
        Returns:
            Dictionary with quality metrics and recommendations
        """
        quality_report = {
            'total_records': len(vessel_h3_data),
            'valid_h3_cells': vessel_h3_data['h3_cell'].notna().sum(),
            'coverage_pct': vessel_h3_data['h3_cell'].notna().mean() * 100
        }
        
        # Time gap analysis
        if 'time_diff_hours' in vessel_h3_data.columns:
            time_gaps = vessel_h3_data['time_diff_hours'].dropna()
            quality_report.update({
                'median_time_gap_hours': time_gaps.median(),
                'large_gaps_count': (time_gaps > 24).sum(),
                'large_gaps_pct': (time_gaps > 24).mean() * 100
            })
        
        # Coordinate validation
        coord_issues = (
            (vessel_h3_data['lat'].abs() > 90) | 
            (vessel_h3_data['lon'].abs() > 180) |
            vessel_h3_data['lat'].isna() |
            vessel_h3_data['lon'].isna()
        ).sum()
        
        quality_report['coordinate_issues'] = coord_issues
        quality_report['coordinate_issues_pct'] = (coord_issues / len(vessel_h3_data)) * 100
        
        # Recommendations
        recommendations = []
        
        if quality_report['coverage_pct'] < 95:
            recommendations.append("Low H3 cell coverage - check coordinate data quality")
        
        if quality_report.get('large_gaps_pct', 0) > 10:
            recommendations.append("Many large time gaps - consider data interpolation")
        
        if quality_report['coordinate_issues_pct'] > 1:
            recommendations.append("Coordinate data issues detected - clean before processing")
        
        if not recommendations:
            recommendations.append("Data quality looks good for H3 feature engineering")
        
        quality_report['recommendations'] = recommendations
        quality_report['overall_quality'] = 'Good' if len(recommendations) == 1 else 'Needs attention'
        
        return quality_report


def batch_process_vessels(vessel_data_dict: Dict[str, pd.DataFrame], 
                         h3_resolution: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Process multiple vessels into H3 sequences
    
    Args:
        vessel_data_dict: Dictionary mapping vessel IDs to DataFrames
        h3_resolution: H3 resolution level
        
    Returns:
        Dictionary mapping vessel IDs to H3 sequence DataFrames
    """
    tracker = VesselH3Tracker(h3_resolution)
    processed_vessels = {}
    
    for vessel_id, vessel_df in vessel_data_dict.items():
        try:
            h3_sequence = tracker.convert_vessel_to_h3_sequence(vessel_df)
            processed_vessels[vessel_id] = h3_sequence
            print(f"Processed vessel {vessel_id}: {len(h3_sequence)} records")
        except Exception as e:
            print(f"Error processing vessel {vessel_id}: {e}")
            continue
    
    return processed_vessels


# Utility functions for H3 operations
def get_h3_neighbors(h3_cell: str, k_rings: int = 1) -> List[str]:
    """Get neighboring H3 cells within k rings"""
    try:
        return list(h3.k_ring(h3_cell, k_rings))
    except:
        return []


def calculate_h3_distance(cell1: str, cell2: str) -> Optional[float]:
    """Calculate distance between two H3 cells in kilometers"""
    try:
        lat1, lon1 = h3.h3_to_geo(cell1)
        lat2, lon2 = h3.h3_to_geo(cell2)
        
        # Haversine distance calculation
        from math import radians, cos, sin, asin, sqrt
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth radius in kilometers
        return c * r
    except:
        return None


if __name__ == "__main__":
    # Example usage
    print("VesselH3Tracker module loaded successfully")
    print("Example usage:")
    print("  tracker = VesselH3Tracker(h3_resolution=5)")
    print("  h3_data = tracker.convert_vessel_to_h3_sequence(vessel_dataframe)")
    print("  analysis = tracker.analyze_vessel_journey(h3_data)")
