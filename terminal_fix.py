# Quick Terminal Discovery Fix
# 
# The issue is that the terminal clustering function expects a 'point_type' column
# but we're creating endpoints with 'endpoint_type'. Let's create a simple fix.

import pandas as pd
import numpy as np
from pathlib import Path

# Create a simple terminal discovery from journey endpoints
def create_simple_terminals_from_journeys(journeys_df, config):
    """Create terminals from journey start/end points with simple clustering"""
    
    endpoints_list = []
    
    for _, journey in journeys_df.iterrows():
        mmsi = journey['mmsi']
        journey_id = journey['journey_id']
        
        # Get start point
        if pd.notna(journey['start_lat']) and pd.notna(journey['start_lon']):
            endpoints_list.append({
                'mmsi': mmsi,
                'journey_id': journey_id,
                'lat': journey['start_lat'],
                'lon': journey['start_lon'],
                'point_type': 'start',  # Use point_type instead of endpoint_type
                'timestamp': journey.get('start_time', None)
            })
        
        # Get end point
        if pd.notna(journey['end_lat']) and pd.notna(journey['end_lon']):
            endpoints_list.append({
                'mmsi': mmsi,
                'journey_id': journey_id,
                'lat': journey['end_lat'],
                'lon': journey['end_lon'],
                'point_type': 'end',  # Use point_type instead of endpoint_type
                'timestamp': journey.get('end_time', None)
            })
    
    endpoints_df = pd.DataFrame(endpoints_list)
    
    if endpoints_df.empty:
        return pd.DataFrame()
    
    # Simple grid-based clustering for terminals
    # Round coordinates to create ~5km grid
    grid_size = 0.05  # ~5km
    
    endpoints_df['lat_grid'] = np.round(endpoints_df['lat'] / grid_size) * grid_size
    endpoints_df['lon_grid'] = np.round(endpoints_df['lon'] / grid_size) * grid_size
    
    # Group by grid to find terminals
    terminal_groups = endpoints_df.groupby(['lat_grid', 'lon_grid']).agg({
        'mmsi': ['count', 'nunique'],
        'journey_id': 'nunique',
        'lat': 'mean',
        'lon': 'mean'
    }).reset_index()
    
    # Flatten column names
    terminal_groups.columns = ['lat_grid', 'lon_grid', 'total_visits', 'unique_vessels', 'unique_journeys', 'centroid_lat', 'centroid_lon']
    
    # Filter for significant terminals
    min_visits = config['terminals']['min_samples']
    min_vessels = config['terminals']['min_vessels']
    
    valid_terminals = terminal_groups[
        (terminal_groups['total_visits'] >= min_visits) &
        (terminal_groups['unique_vessels'] >= min_vessels)
    ].copy()
    
    if not valid_terminals.empty:
        valid_terminals['terminal_id'] = range(1, len(valid_terminals) + 1)
        valid_terminals = valid_terminals.reindex(columns=[
            'terminal_id', 'centroid_lat', 'centroid_lon', 
            'total_visits', 'unique_vessels', 'unique_journeys'
        ])
    
    return valid_terminals

print("Terminal discovery fix functions created")
