"""
Functions for clustering trajectories and terminal points.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import logging
from pathlib import Path
from typing import Tuple, List
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw
from shapely.geometry import Point

def extract_journey_endpoints(journeys_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts all start and end points from journey sequences.

    Args:
        journeys_df (pd.DataFrame): DataFrame with journey data including start/end coordinates.

    Returns:
        pd.DataFrame: DataFrame with all endpoint coordinates and metadata.
    """
    endpoints = []
    
    for _, journey in journeys_df.iterrows():
        # Start point
        endpoints.append({
            'mmsi': journey['mmsi'],
            'journey_id': journey['journey_id'],
            'point_type': 'start',
            'lat': journey['start_lat'],
            'lon': journey['start_lon'],
            'timestamp': journey['start_time']
        })
        
        # End point
        endpoints.append({
            'mmsi': journey['mmsi'],
            'journey_id': journey['journey_id'],
            'point_type': 'end',
            'lat': journey['end_lat'],
            'lon': journey['end_lon'],
            'timestamp': journey['end_time']
        })
    
    endpoints_df = pd.DataFrame(endpoints)
    logging.info(f"Extracted {len(endpoints_df)} journey endpoints")
    return endpoints_df

def cluster_terminal_points(journey_endpoints: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clusters journey start/end points to identify terminals using DBSCAN.

    Args:
        journey_endpoints (pd.DataFrame): DataFrame with lat/lon of journey start and end points.
        config (dict): Configuration with DBSCAN parameters (eps, min_samples).

    Returns:
        pd.DataFrame: The input DataFrame with an added 'terminal_id' column.
    """
    if journey_endpoints.empty:
        logging.warning("No journey endpoints to cluster")
        return journey_endpoints
    
    # Prepare coordinates for clustering
    coords = journey_endpoints[['lat', 'lon']].values
    
    # Apply DBSCAN clustering
    eps = config.get('eps', 0.1)  # ~10km at equator
    min_samples = config.get('min_samples', 5)
    
    # Use geographic distance for clustering (degrees)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = clustering.fit_predict(coords)
    
    # Add cluster labels to dataframe
    journey_endpoints = journey_endpoints.copy()
    journey_endpoints['terminal_id'] = cluster_labels
    
    # Count clusters and noise points
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    logging.info(f"Found {n_clusters} terminal clusters with {n_noise} noise points")
    
    return journey_endpoints

def create_terminal_summary(clustered_endpoints: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary of terminal clusters with centroid locations and metadata.

    Args:
        clustered_endpoints (pd.DataFrame): DataFrame with clustered endpoint data.

    Returns:
        pd.DataFrame: Summary DataFrame with terminal information.
    """
    # Filter out noise points (terminal_id = -1)
    valid_terminals = clustered_endpoints[clustered_endpoints['terminal_id'] >= 0]
    
    if valid_terminals.empty:
        logging.warning("No valid terminal clusters found")
        return pd.DataFrame()
    
    terminal_summary = []
    
    for terminal_id, group in valid_terminals.groupby('terminal_id'):
        # Calculate centroid
        centroid_lat = group['lat'].mean()
        centroid_lon = group['lon'].mean()
        
        # Count activity
        total_visits = len(group)
        unique_vessels = group['mmsi'].nunique()
        start_visits = len(group[group['point_type'] == 'start'])
        end_visits = len(group[group['point_type'] == 'end'])
        
        # Time range
        first_visit = group['timestamp'].min()
        last_visit = group['timestamp'].max()
        
        terminal_summary.append({
            'terminal_id': terminal_id,
            'centroid_lat': centroid_lat,
            'centroid_lon': centroid_lon,
            'total_visits': total_visits,
            'unique_vessels': unique_vessels,
            'start_visits': start_visits,
            'end_visits': end_visits,
            'first_visit': first_visit,
            'last_visit': last_visit,
            'lat_std': group['lat'].std(),
            'lon_std': group['lon'].std()
        })
    
    summary_df = pd.DataFrame(terminal_summary)
    logging.info(f"Created summary for {len(summary_df)} terminals")
    return summary_df

def save_terminals_to_geopackage(terminal_summary: pd.DataFrame, output_path: str) -> None:
    """
    Saves terminal data to a GeoPackage file.

    Args:
        terminal_summary (pd.DataFrame): Summary of terminal clusters.
        output_path (str): Path to save the GeoPackage file.
    """
    if terminal_summary.empty:
        logging.warning("No terminal data to save")
        return
    
    # Create geometry column
    geometry = [Point(lon, lat) for lat, lon in 
                zip(terminal_summary['centroid_lat'], terminal_summary['centroid_lon'])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(terminal_summary, geometry=geometry, crs='EPSG:4326')
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to GeoPackage
    gdf.to_file(output_path, driver='GPKG')
    logging.info(f"Saved {len(gdf)} terminals to {output_path}")

def compute_dtw_distance_matrix(h3_sequences: list) -> np.ndarray:
    """
    Computes the pairwise Dynamic Time Warping (DTW) distance between all sequences.

    Args:
        h3_sequences (list): A list of H3 sequences (which are lists of H3 indices).

    Returns:
        np.ndarray: A square matrix of pairwise distances.
    """
    n_sequences = len(h3_sequences)
    distance_matrix = np.zeros((n_sequences, n_sequences))
    
    logging.info(f"Computing DTW distance matrix for {n_sequences} sequences")
    
    for i in range(n_sequences):
        if i % 100 == 0:
            logging.info(f"Processing sequence {i+1}/{n_sequences}")
        
        for j in range(i+1, n_sequences):
            # Convert H3 sequences to numeric for DTW
            seq1 = [hash(cell) % 1000000 for cell in h3_sequences[i]]  # Convert to numeric
            seq2 = [hash(cell) % 1000000 for cell in h3_sequences[j]]
            
            # Compute DTW distance
            distance = dtw.distance(seq1, seq2)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix
    
    logging.info("DTW distance matrix computation completed")
    return distance_matrix

def cluster_routes(distance_matrix: np.ndarray, config: dict) -> Tuple[np.ndarray, int, int]:
    """
    Clusters routes using DBSCAN on a pre-computed distance matrix.

    Args:
        distance_matrix (np.ndarray): The pairwise DTW distance matrix.
        config (dict): Configuration with DBSCAN parameters (eps, min_samples).

    Returns:
        tuple: A tuple containing (cluster_labels, number_of_clusters, number_of_outliers).
    """
    eps = config.get('eps', 2.0)
    min_samples = config.get('min_samples', 3)
    
    # Use precomputed distance matrix with DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Count clusters and outliers
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = list(cluster_labels).count(-1)
    
    logging.info(f"Route clustering found {n_clusters} clusters with {n_outliers} outliers")
    
    return cluster_labels, n_clusters, n_outliers

def assign_route_clusters_to_journeys(journeys_df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Assigns route cluster IDs back to the journey data.

    Args:
        journeys_df (pd.DataFrame): Original journey data.
        cluster_labels (np.ndarray): Cluster labels from DBSCAN.

    Returns:
        pd.DataFrame: Journey data with added 'route_cluster_id' column.
    """
    journeys_with_clusters = journeys_df.copy()
    journeys_with_clusters['route_cluster_id'] = cluster_labels
    
    logging.info(f"Assigned route clusters to {len(journeys_with_clusters)} journeys")
    return journeys_with_clusters
