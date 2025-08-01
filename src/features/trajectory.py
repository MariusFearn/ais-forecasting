"""
Functions for trajectory extraction, segmentation, and analysis.
"""
import pandas as pd
import numpy as np
import h3
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from ..data.loader import AISDataLoader
from ..data.duckdb_engine import DuckDBEngine

def load_vessel_data(mmsi: str, data_loader: AISDataLoader) -> pd.DataFrame:
    """
    Load data for a single vessel using existing data loading infrastructure.

    Args:
        mmsi (str): The MMSI identifier for the vessel.
        data_loader (AISDataLoader): Configured data loader instance.

    Returns:
        pd.DataFrame: DataFrame with vessel's AIS data, sorted by timestamp.
    """
    try:
        # Use DuckDB engine for efficient single vessel loading
        query = f"""
        SELECT mmsi, timestamp, lat, lon, sog, cog
        FROM ais_data 
        WHERE mmsi = '{mmsi}'
        ORDER BY timestamp ASC
        """
        
        if data_loader.use_duckdb:
            df = data_loader.duckdb_engine.execute_query(query)
        else:
            # Fallback to pandas loading if DuckDB not available
            df = data_loader.load_raw_data()
            df = df[df['mmsi'] == mmsi].sort_values('timestamp')
        
        logging.info(f"Loaded {len(df)} records for vessel {mmsi}")
        return df
        
    except Exception as e:
        logging.error(f"Error loading data for vessel {mmsi}: {e}")
        return pd.DataFrame()

def segment_into_journeys(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Segments a vessel's AIS data into distinct journeys based on time gaps.

    Args:
        df (pd.DataFrame): DataFrame with a single vessel's data, sorted by timestamp.
        config (dict): Configuration dictionary with parameters like 'time_gap_threshold_hours'.

    Returns:
        pd.DataFrame: A DataFrame with an added 'journey_id' column.
    """
    if df.empty:
        return df
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate time differences between consecutive points
    time_diffs = df['timestamp'].diff()
    
    # Find gaps larger than threshold
    threshold_hours = config.get('time_gap_threshold_hours', 24)
    threshold = pd.Timedelta(hours=threshold_hours)
    
    # Create journey breaks where time gap exceeds threshold
    journey_breaks = time_diffs > threshold
    
    # Assign journey IDs
    df['journey_id'] = journey_breaks.cumsum()
    
    # Filter out journeys that are too short
    min_length = config.get('min_journey_length', 10)
    journey_counts = df['journey_id'].value_counts()
    valid_journeys = journey_counts[journey_counts >= min_length].index
    df = df[df['journey_id'].isin(valid_journeys)]
    
    # Reassign sequential journey IDs
    journey_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(df['journey_id'].unique()))}
    df['journey_id'] = df['journey_id'].map(journey_mapping)
    
    logging.info(f"Segmented into {len(df['journey_id'].unique())} journeys")
    return df

def journeys_to_h3_sequences(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Converts journeys into sequences of H3 cell indices.

    Args:
        df (pd.DataFrame): DataFrame with journey IDs.
        config (dict): Configuration dictionary with H3 resolution.

    Returns:
        pd.DataFrame: A DataFrame of (mmsi, journey_id, h3_sequence).
    """
    if df.empty:
        return pd.DataFrame(columns=['mmsi', 'journey_id', 'h3_sequence'])
    
    h3_resolution = config.get('h3_resolution', 5)
    
    # Convert lat/lon to H3 cells
    df['h3_cell'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], h3_resolution), axis=1)
    
    # Group by mmsi and journey_id to create sequences
    sequences = []
    for (mmsi, journey_id), group in df.groupby(['mmsi', 'journey_id']):
        h3_sequence = group['h3_cell'].tolist()
        sequences.append({
            'mmsi': mmsi,
            'journey_id': journey_id,
            'h3_sequence': h3_sequence,
            'start_time': group['timestamp'].min(),
            'end_time': group['timestamp'].max(),
            'start_lat': group.iloc[0]['lat'],
            'start_lon': group.iloc[0]['lon'],
            'end_lat': group.iloc[-1]['lat'],
            'end_lon': group.iloc[-1]['lon']
        })
    
    result_df = pd.DataFrame(sequences)
    logging.info(f"Created {len(result_df)} H3 journey sequences")
    return result_df

def process_all_vessels(config: dict) -> pd.DataFrame:
    """
    Main function to process all vessels and save journeys to a Parquet file.

    Args:
        config (dict): Configuration dictionary with data paths and parameters.

    Returns:
        pd.DataFrame: DataFrame of all journey sequences.
    """
    # Initialize data loader
    data_dir = config['data']['raw_data_dir']
    data_loader = AISDataLoader(data_dir, use_duckdb=True)
    
    # Get list of all unique vessels
    try:
        if data_loader.use_duckdb:
            mmsi_query = "SELECT DISTINCT mmsi FROM ais_data"
            mmsi_df = data_loader.duckdb_engine.execute_query(mmsi_query)
            unique_vessels = mmsi_df['mmsi'].tolist()
        else:
            # Fallback method
            all_data = data_loader.load_raw_data()
            unique_vessels = all_data['mmsi'].unique().tolist()
        
        logging.info(f"Found {len(unique_vessels)} unique vessels to process")
        
    except Exception as e:
        logging.error(f"Error getting vessel list: {e}")
        return pd.DataFrame()
    
    # Process each vessel
    all_journeys = []
    for i, mmsi in enumerate(unique_vessels):
        if i % 100 == 0:
            logging.info(f"Processing vessel {i+1}/{len(unique_vessels)}: {mmsi}")
        
        # Load vessel data
        vessel_df = load_vessel_data(mmsi, data_loader)
        if vessel_df.empty:
            continue
        
        # Segment into journeys
        segmented_df = segment_into_journeys(vessel_df, config['trajectory'])
        if segmented_df.empty:
            continue
        
        # Convert to H3 sequences
        journey_sequences = journeys_to_h3_sequences(segmented_df, config)
        if not journey_sequences.empty:
            all_journeys.append(journey_sequences)
    
    # Combine all results
    if all_journeys:
        final_df = pd.concat(all_journeys, ignore_index=True)
        
        # Save to output file
        output_path = Path(config['trajectory']['output_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_path, index=False)
        
        logging.info(f"Saved {len(final_df)} journey sequences to {output_path}")
        return final_df
    else:
        logging.warning("No valid journeys found")
        return pd.DataFrame()

def calculate_route_centroids(clustered_journeys: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the representative centroid for each route cluster.

    Args:
        clustered_journeys (pd.DataFrame): DataFrame of journeys with a 'route_cluster_id'.

    Returns:
        pd.DataFrame: A DataFrame of (cluster_id, centroid_h3_sequence).
    """
    # Filter out noise/outlier routes (cluster_id = -1)
    valid_routes = clustered_journeys[clustered_journeys['route_cluster_id'] >= 0]
    
    if valid_routes.empty:
        logging.warning("No valid route clusters found for centroid calculation")
        return pd.DataFrame()
    
    centroids = []
    
    for cluster_id, group in valid_routes.groupby('route_cluster_id'):
        logging.info(f"Calculating centroid for route cluster {cluster_id} ({len(group)} journeys)")
        
        # Get all H3 sequences in this cluster
        sequences = group['h3_sequence'].tolist()
        
        # Simple approach: find the sequence closest to the "center" of the cluster
        # by finding the sequence with minimum average DTW distance to all others
        if len(sequences) == 1:
            centroid_sequence = sequences[0]
        else:
            # Calculate the medoid (most representative sequence)
            centroid_sequence = _calculate_sequence_medoid(sequences)
        
        # Calculate cluster metadata
        total_journeys = len(group)
        unique_vessels = group['mmsi'].nunique()
        avg_duration = (group['end_time'] - group['start_time']).mean()
        
        centroids.append({
            'route_cluster_id': cluster_id,
            'centroid_h3_sequence': centroid_sequence,
            'total_journeys': total_journeys,
            'unique_vessels': unique_vessels,
            'avg_duration_hours': avg_duration.total_seconds() / 3600 if pd.notna(avg_duration) else None,
            'first_journey': group['start_time'].min(),
            'last_journey': group['end_time'].max()
        })
    
    result_df = pd.DataFrame(centroids)
    logging.info(f"Calculated centroids for {len(result_df)} route clusters")
    return result_df

def _calculate_sequence_medoid(sequences: List[List[str]]) -> List[str]:
    """
    Finds the medoid (most representative) sequence from a list of H3 sequences.
    
    Args:
        sequences (List[List[str]]): List of H3 cell sequences.
        
    Returns:
        List[str]: The medoid sequence.
    """
    from dtaidistance import dtw
    
    if len(sequences) <= 3:
        # For small clusters, just return the first sequence
        return sequences[0]
    
    # Convert H3 to numeric for DTW computation
    numeric_sequences = []
    for seq in sequences:
        numeric_seq = [hash(cell) % 1000000 for cell in seq]
        numeric_sequences.append(numeric_seq)
    
    # Calculate sum of distances for each sequence to all others
    min_total_distance = float('inf')
    medoid_idx = 0
    
    for i, seq1 in enumerate(numeric_sequences):
        total_distance = 0
        for j, seq2 in enumerate(numeric_sequences):
            if i != j:
                total_distance += dtw.distance(seq1, seq2)
        
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            medoid_idx = i
    
    return sequences[medoid_idx]

def link_routes_to_terminals(route_centroids: pd.DataFrame, 
                           terminal_summary: pd.DataFrame,
                           clustered_journeys: pd.DataFrame) -> pd.DataFrame:
    """
    Links each route centroid to its most likely start and end terminals.

    Args:
        route_centroids (pd.DataFrame): Route centroid data.
        terminal_summary (pd.DataFrame): Terminal cluster summary.
        clustered_journeys (pd.DataFrame): Original journey data with cluster assignments.

    Returns:
        pd.DataFrame: Routes with terminal connections and metadata.
    """
    if route_centroids.empty or terminal_summary.empty:
        logging.warning("Missing data for route-terminal linking")
        return pd.DataFrame()
    
    routes_with_terminals = []
    
    for _, route in route_centroids.iterrows():
        cluster_id = route['route_cluster_id']
        
        # Get all journeys in this route cluster
        route_journeys = clustered_journeys[
            clustered_journeys['route_cluster_id'] == cluster_id
        ]
        
        if route_journeys.empty:
            continue
        
        # Find most common start and end terminals for this route
        start_terminal = _find_most_common_terminal(
            route_journeys, terminal_summary, 'start'
        )
        end_terminal = _find_most_common_terminal(
            route_journeys, terminal_summary, 'end'
        )
        
        # Convert H3 sequence to geographic coordinates for the route line
        route_coordinates = _h3_sequence_to_coordinates(route['centroid_h3_sequence'])
        
        routes_with_terminals.append({
            'route_id': cluster_id,
            'start_terminal_id': start_terminal,
            'end_terminal_id': end_terminal,
            'route_coordinates': route_coordinates,
            'total_journeys': route['total_journeys'],
            'unique_vessels': route['unique_vessels'],
            'avg_duration_hours': route['avg_duration_hours'],
            'first_journey': route['first_journey'],
            'last_journey': route['last_journey'],
            'h3_sequence': route['centroid_h3_sequence']
        })
    
    result_df = pd.DataFrame(routes_with_terminals)
    logging.info(f"Linked {len(result_df)} routes to terminals")
    return result_df

def _find_most_common_terminal(route_journeys: pd.DataFrame, 
                             terminal_summary: pd.DataFrame, 
                             endpoint_type: str) -> int:
    """
    Finds the most common terminal for route start or end points.
    
    Args:
        route_journeys (pd.DataFrame): Journeys in a specific route cluster.
        terminal_summary (pd.DataFrame): Terminal cluster information.
        endpoint_type (str): 'start' or 'end'.
        
    Returns:
        int: Terminal ID or -1 if no clear terminal found.
    """
    # Get coordinates based on endpoint type
    if endpoint_type == 'start':
        coords = route_journeys[['start_lat', 'start_lon']].values
    else:
        coords = route_journeys[['end_lat', 'end_lon']].values
    
    # Find closest terminal for each journey endpoint
    terminal_votes = []
    
    for coord in coords:
        lat, lon = coord
        # Calculate distances to all terminal centroids
        distances = []
        for _, terminal in terminal_summary.iterrows():
            dist = ((terminal['centroid_lat'] - lat) ** 2 + 
                   (terminal['centroid_lon'] - lon) ** 2) ** 0.5
            distances.append((terminal['terminal_id'], dist))
        
        # Vote for the closest terminal
        if distances:
            closest_terminal = min(distances, key=lambda x: x[1])[0]
            terminal_votes.append(closest_terminal)
    
    # Return the most common terminal
    if terminal_votes:
        from collections import Counter
        most_common = Counter(terminal_votes).most_common(1)[0][0]
        return most_common
    else:
        return -1

def _h3_sequence_to_coordinates(h3_sequence: List[str]) -> List[Tuple[float, float]]:
    """
    Converts an H3 sequence to lat/lon coordinates.
    
    Args:
        h3_sequence (List[str]): Sequence of H3 cell indices.
        
    Returns:
        List[Tuple[float, float]]: List of (lat, lon) coordinates.
    """
    coordinates = []
    for h3_cell in h3_sequence:
        lat, lon = h3.h3_to_geo(h3_cell)
        coordinates.append((lat, lon))
    
    return coordinates

def save_route_graph_to_geopackage(routes_with_terminals: pd.DataFrame, 
                                 output_path: str) -> None:
    """
    Saves the final route graph (edges) to a GeoPackage file.

    Args:
        routes_with_terminals (pd.DataFrame): Route data with terminal connections.
        output_path (str): Path to save the GeoPackage file.
    """
    if routes_with_terminals.empty:
        logging.warning("No route data to save")
        return
    
    import geopandas as gpd
    from shapely.geometry import LineString
    
    # Create LineString geometries from route coordinates
    geometries = []
    for _, route in routes_with_terminals.iterrows():
        coords = route['route_coordinates']
        if len(coords) >= 2:
            # Convert (lat, lon) to (lon, lat) for Shapely
            shapely_coords = [(lon, lat) for lat, lon in coords]
            line = LineString(shapely_coords)
            geometries.append(line)
        else:
            geometries.append(None)
    
    # Create GeoDataFrame
    routes_gdf = routes_with_terminals.copy()
    routes_gdf['geometry'] = geometries
    routes_gdf = gpd.GeoDataFrame(routes_gdf, crs='EPSG:4326')
    
    # Remove the coordinate list column since we have geometry now
    routes_gdf = routes_gdf.drop('route_coordinates', axis=1)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to GeoPackage
    routes_gdf.to_file(output_path, driver='GPKG')
    logging.info(f"Saved {len(routes_gdf)} routes to {output_path}")
