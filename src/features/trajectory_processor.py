"""Optimized trajectory processing using original AIS column names."""

import h3
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..utils.logging_setup import get_logger, log_performance_metrics

logger = get_logger(__name__)

def extract_vessel_trajectories(
    ais_data: pd.DataFrame,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Extract vessel trajectories with H3 spatial indexing using original column names.
    
    Uses original AIS column names:
    - imo: vessel identifier (not mmsi)  
    - mdt: timestamp (not timestamp)
    - lat: latitude
    - lon: longitude
    
    Args:
        ais_data: AIS DataFrame with columns [imo, mdt, lat, lon, speed]
        config: Configuration dictionary with trajectory parameters
        
    Returns:
        DataFrame with processed trajectories containing H3 sequences
        
    Raises:
        ValueError: If input data is invalid or empty
    """
    if config is None:
        config = {
            'h3_resolution': 5,
            'min_journey_length': 5,
            'speed_threshold_knots': 0.5,
            'time_gap_hours': 6
        }
    
    logger.info(f"Extracting vessel trajectories with H3 resolution {config.get('h3_resolution', 5)}")
    start_time = time.time()
    
    # Validate input data
    required_columns = ['imo', 'lat', 'lon', 'mdt', 'speed']
    missing_columns = [col for col in required_columns if col not in ais_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if ais_data.empty:
        logger.warning("Empty input data, returning empty DataFrame")
        return pd.DataFrame()
    
    logger.info(f"Processing {len(ais_data):,} records from {ais_data['imo'].nunique():,} vessels")
    
    trajectories = []
    
    # Process each vessel separately
    for vessel_id in ais_data['imo'].unique():
        vessel_data = ais_data[ais_data['imo'] == vessel_id].copy()
        vessel_data = vessel_data.sort_values('mdt').reset_index(drop=True)
        
        if len(vessel_data) < config.get('min_journey_length', 5):
            continue
            
        # Add H3 cells
        vessel_data['h3_cell'] = vessel_data.apply(
            lambda row: h3.geo_to_h3(row['lat'], row['lon'], config.get('h3_resolution', 5)), 
            axis=1
        )
        
        # Split into trajectory segments based on time gaps
        vessel_data['time_diff'] = vessel_data['mdt'].diff().dt.total_seconds() / 3600  # hours
        gap_indices = vessel_data[vessel_data['time_diff'] > config.get('time_gap_hours', 6)].index
        
        # Create trajectory segments
        start_idx = 0
        for gap_idx in list(gap_indices) + [len(vessel_data)]:
            segment = vessel_data.iloc[start_idx:gap_idx].copy()
            
            if len(segment) >= config.get('min_journey_length', 5):
                # Create trajectory summary
                trajectory = {
                    'vessel_id': vessel_id,
                    'trajectory_id': f"{vessel_id}_{start_idx}",
                    'start_time': segment['mdt'].min(),
                    'end_time': segment['mdt'].max(),
                    'duration_hours': (segment['mdt'].max() - segment['mdt'].min()).total_seconds() / 3600,
                    'point_count': len(segment),
                    'unique_h3_cells': segment['h3_cell'].nunique(),
                    'avg_speed': segment['speed'].mean(),
                    'max_speed': segment['speed'].max(),
                    'stationary_points': len(segment[segment['speed'] < config.get('speed_threshold_knots', 0.5)]),
                    'h3_sequence': segment['h3_cell'].tolist(),
                    'start_lat': segment['lat'].iloc[0],
                    'start_lon': segment['lon'].iloc[0],
                    'end_lat': segment['lat'].iloc[-1],
                    'end_lon': segment['lon'].iloc[-1]
                }
                trajectories.append(trajectory)
            
            start_idx = gap_idx
    
    result_df = pd.DataFrame(trajectories)
    
    processing_time = time.time() - start_time
    logger.info(f"Extracted {len(result_df)} trajectory segments in {processing_time:.2f} seconds")
    
    return result_df
    
    if ais_data.empty:
        raise ValueError("Input AIS data is empty")
    
    # Validate required columns (original names)
    required_columns = ['imo', 'mdt', 'lat', 'lon']
    missing_columns = [col for col in required_columns if col not in ais_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Sort data by vessel and time
    logger.info("Sorting AIS data by vessel and timestamp")
    ais_data = ais_data.sort_values(['imo', 'mdt']).reset_index(drop=True)
    
    # Group by vessel for processing
    vessel_groups = list(ais_data.groupby('imo'))
    logger.info(f"Processing trajectories for {len(vessel_groups)} unique vessels")
    
    if use_multiprocessing and len(vessel_groups) > 10:
        # Parallel processing for large datasets
        journeys = _process_trajectories_parallel(
            vessel_groups, h3_resolution, min_journey_length, 
            max_journey_length, speed_threshold_knots, time_gap_hours
        )
    else:
        # Sequential processing for small datasets
        journeys = _process_trajectories_sequential(
            vessel_groups, h3_resolution, min_journey_length,
            max_journey_length, speed_threshold_knots, time_gap_hours
        )
    
    # Convert to DataFrame
    if journeys:
        journeys_df = pd.DataFrame(journeys)
        
        # Optimize H3 sequences
        logger.info("Optimizing H3 sequences")
        journeys_df['h3_sequence'] = journeys_df['h3_sequence'].apply(optimize_h3_sequences)
        
        # Final filtering by length
        valid_journeys = (
            (journeys_df['h3_sequence'].str.len() >= min_journey_length) &
            (journeys_df['h3_sequence'].str.len() <= max_journey_length)
        )
        journeys_df = journeys_df[valid_journeys].reset_index(drop=True)
        
        duration = time.time() - start_time
        log_performance_metrics(
            logger, 
            "Trajectory Extraction", 
            duration, 
            len(journeys_df)
        )
        
        logger.info(f"Successfully extracted {len(journeys_df):,} vessel trajectories")
        return journeys_df
    
    else:
        logger.warning("No valid trajectories found")
        return pd.DataFrame()

def _process_trajectories_parallel(
    vessel_groups: List[Tuple],
    h3_resolution: int,
    min_journey_length: int,
    max_journey_length: int,
    speed_threshold_knots: float,
    time_gap_hours: int
) -> List[Dict]:
    """Process vessel trajectories using parallel processing."""
    
    logger.info("Using parallel processing for trajectory extraction")
    
    # Determine optimal number of processes
    n_processes = min(mp.cpu_count(), len(vessel_groups), 8)  # Cap at 8 processes
    
    all_journeys = []
    
    with ProcessPoolExecutor(max_processes=n_processes) as executor:
        # Submit tasks
        future_to_mmsi = {
            executor.submit(
                process_trajectories_batch,
                vessel_group,
                h3_resolution,
                min_journey_length,
                max_journey_length,
                speed_threshold_knots,
                time_gap_hours
            ): mmsi for mmsi, vessel_group in vessel_groups
        }
        
        # Collect results
        completed = 0
        for future in as_completed(future_to_mmsi):
            mmsi = future_to_mmsi[future]
            try:
                vessel_journeys = future.result()
                all_journeys.extend(vessel_journeys)
                completed += 1
                
                if completed % 100 == 0:
                    logger.debug(f"Processed {completed}/{len(vessel_groups)} vessels")
                    
            except Exception as e:
                logger.error(f"Error processing vessel {mmsi}: {e}")
    
    return all_journeys

def _process_trajectories_sequential(
    vessel_groups: List[Tuple],
    h3_resolution: int,
    min_journey_length: int,
    max_journey_length: int,
    speed_threshold_knots: float,
    time_gap_hours: int
) -> List[Dict]:
    """Process vessel trajectories sequentially."""
    
    logger.info("Using sequential processing for trajectory extraction")
    
    all_journeys = []
    
    for i, (mmsi, vessel_group) in enumerate(vessel_groups):
        try:
            vessel_journeys = process_trajectories_batch(
                vessel_group,
                h3_resolution,
                min_journey_length,
                max_journey_length,
                speed_threshold_knots,
                time_gap_hours
            )
            all_journeys.extend(vessel_journeys)
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Processed {i + 1}/{len(vessel_groups)} vessels")
                
        except Exception as e:
            logger.error(f"Error processing vessel {mmsi}: {e}")
    
    return all_journeys

def process_trajectories_batch(
    vessel_group: pd.DataFrame,
    h3_resolution: int,
    min_journey_length: int = 5,
    max_journey_length: int = 1000,
    speed_threshold_knots: float = 0.5,
    time_gap_hours: int = 6
) -> List[Dict]:
    """
    Process trajectories for a batch of vessels using original column names.
    
    Args:
        vessel_group: DataFrame containing AIS data for a single vessel [imo, mdt, lat, lon]
        h3_resolution: H3 spatial resolution
        min_journey_length: Minimum journey length
        max_journey_length: Maximum journey length
        speed_threshold_knots: Speed threshold for movement detection
        time_gap_hours: Time gap threshold for journey segmentation
        
    Returns:
        List of journey dictionaries
    """
    if len(vessel_group) < min_journey_length:
        return []
    
    imo = vessel_group['imo'].iloc[0]  # Use imo instead of mmsi
    journeys = []
    
    # Convert coordinates to H3
    h3_indices = []
    for _, row in vessel_group.iterrows():
        try:
            h3_index = h3.geo_to_h3(row['lat'], row['lon'], h3_resolution)  # Use lat/lon
            h3_indices.append(h3_index)
        except Exception:
            h3_indices.append(None)
    
    vessel_group = vessel_group.copy()
    vessel_group['h3_index'] = h3_indices
    
    # Remove rows with invalid H3 indices
    vessel_group = vessel_group[vessel_group['h3_index'].notna()].copy()
    
    if len(vessel_group) < min_journey_length:
        return []
    
    # Calculate time differences
    vessel_group['time_diff_hours'] = vessel_group['mdt'].diff().dt.total_seconds() / 3600  # Use mdt
    
    # Identify journey breaks (large time gaps)
    journey_breaks = vessel_group['time_diff_hours'] > time_gap_hours
    vessel_group['journey_id'] = journey_breaks.cumsum()
    
    # Process each journey segment
    for journey_id, journey_data in vessel_group.groupby('journey_id'):
        if len(journey_data) >= min_journey_length:
            
            # Create H3 sequence
            h3_sequence = journey_data['h3_index'].tolist()
            
            # Skip if too long (performance)
            if len(h3_sequence) > max_journey_length:
                continue
            
            journey_dict = {
                'imo': imo,  # Use imo instead of mmsi
                'journey_id': f"{imo}_{journey_id}",
                'start_time': journey_data['mdt'].min(),  # Use mdt
                'end_time': journey_data['mdt'].max(),    # Use mdt
                'start_lat': journey_data['lat'].iloc[0], # Use lat
                'start_lon': journey_data['lon'].iloc[0], # Use lon
                'end_lat': journey_data['lat'].iloc[-1],  # Use lat
                'end_lon': journey_data['lon'].iloc[-1],  # Use lon
                'h3_sequence': h3_sequence,
                'duration_hours': (journey_data['mdt'].max() -   # Use mdt
                                 journey_data['mdt'].min()).total_seconds() / 3600,
                'num_points': len(journey_data)
            }
            
            journeys.append(journey_dict)
    
    return journeys

def optimize_h3_sequences(h3_sequence: List[str]) -> List[str]:
    """
    Optimize H3 sequences for further processing by removing duplicates and smoothing.
    
    Args:
        h3_sequence: List of H3 indices representing a vessel trajectory
        
    Returns:
        Optimized H3 sequence
    """
    if not h3_sequence:
        return h3_sequence
    
    # Remove consecutive duplicates while preserving order
    optimized = [h3_sequence[0]]
    for h3_index in h3_sequence[1:]:
        if h3_index != optimized[-1]:
            optimized.append(h3_index)
    
    return optimized

def calculate_trajectory_metrics(journeys_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for extracted trajectories.
    
    Args:
        journeys_df: DataFrame containing journey data with imo column
        
    Returns:
        Dictionary of trajectory metrics
    """
    if journeys_df.empty:
        return {}
    
    metrics = {
        'total_journeys': len(journeys_df),
        'unique_vessels': journeys_df['imo'].nunique(),  # Use imo instead of mmsi
        'avg_journey_length': journeys_df['h3_sequence'].str.len().mean(),
        'avg_duration_hours': journeys_df['duration_hours'].mean(),
        'avg_points_per_journey': journeys_df['num_points'].mean(),
        'total_h3_points': journeys_df['h3_sequence'].str.len().sum()
    }
    
    return metrics
