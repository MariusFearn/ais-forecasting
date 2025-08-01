"""
Main script to discover shipping lanes from AIS data.

This script orchestrates the entire lane discovery pipeline:
1.  Loads configuration.
2.  Extracts vessel trajectories and identifies terminals.
3.  Clusters trajectories to find common routes.
4.  Constructs a route graph with terminals as nodes and lanes as edges.
5.  Validates and visualizes the final shipping lanes.
"""
import argparse
import logging
import yaml
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.features.trajectory import (
    process_all_vessels, 
    calculate_route_centroids,
    link_routes_to_terminals,
    save_route_graph_to_geopackage
)
from src.models.clustering import (
    extract_journey_endpoints,
    cluster_terminal_points,
    create_terminal_summary,
    save_terminals_to_geopackage,
    compute_dtw_distance_matrix,
    cluster_routes,
    assign_route_clusters_to_journeys
)
from src.visualization.lanes import create_and_save_shipping_lanes_map
from src.utils.metrics import (
    calculate_clustering_metrics,
    calculate_terminal_metrics,
    calculate_route_metrics,
    log_validation_summary
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)

def phase1_trajectory_and_terminal_extraction(config: dict) -> tuple:
    """
    Phase 1: Extract trajectories and identify terminals.
    
    Returns:
        tuple: (journeys_df, terminal_summary_df, clustered_endpoints_df)
    """
    logging.info("=" * 60)
    logging.info("PHASE 1: TRAJECTORY & TERMINAL EXTRACTION")
    logging.info("=" * 60)
    
    # Step 1: Process all vessels and extract trajectories
    logging.info("Step 1: Processing vessel trajectories...")
    journeys_df = process_all_vessels(config)
    
    if journeys_df.empty:
        logging.error("No journey data generated. Exiting.")
        sys.exit(1)
    
    # Step 2: Extract journey endpoints
    logging.info("Step 2: Extracting journey endpoints...")
    endpoints_df = extract_journey_endpoints(journeys_df)
    
    # Step 3: Cluster endpoints to find terminals
    logging.info("Step 3: Clustering endpoints to find terminals...")
    clustered_endpoints = cluster_terminal_points(endpoints_df, config['terminals'])
    
    # Step 4: Create terminal summary
    logging.info("Step 4: Creating terminal summary...")
    terminal_summary = create_terminal_summary(clustered_endpoints)
    
    # Step 5: Save terminals to GeoPackage
    logging.info("Step 5: Saving terminals to GeoPackage...")
    save_terminals_to_geopackage(terminal_summary, config['terminals']['output_path'])
    
    logging.info(f"Phase 1 complete: {len(journeys_df)} journeys, {len(terminal_summary)} terminals")
    return journeys_df, terminal_summary, clustered_endpoints

def phase2_route_clustering(journeys_df, config: dict):
    """
    Phase 2: Cluster trajectories to find common routes.
    
    Returns:
        tuple: (clustered_journeys_df, distance_matrix, cluster_labels)
    """
    logging.info("=" * 60)
    logging.info("PHASE 2: ROUTE CLUSTERING")
    logging.info("=" * 60)
    
    # Step 1: Compute DTW distance matrix
    logging.info("Step 1: Computing DTW distance matrix...")
    h3_sequences = journeys_df['h3_sequence'].tolist()
    distance_matrix = compute_dtw_distance_matrix(h3_sequences)
    
    # Step 2: Cluster routes using DBSCAN
    logging.info("Step 2: Clustering routes...")
    cluster_labels, n_clusters, n_outliers = cluster_routes(distance_matrix, config['routes'])
    
    # Step 3: Assign cluster labels back to journey data
    logging.info("Step 3: Assigning cluster labels to journeys...")
    clustered_journeys = assign_route_clusters_to_journeys(journeys_df, cluster_labels)
    
    # Step 4: Save clustered journeys
    logging.info("Step 4: Saving clustered journeys...")
    output_path = Path(config['routes']['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clustered_journeys.to_parquet(output_path, index=False)
    
    logging.info(f"Phase 2 complete: {n_clusters} route clusters, {n_outliers} outliers")
    return clustered_journeys, distance_matrix, cluster_labels

def phase3_route_graph_construction(clustered_journeys, terminal_summary, config: dict):
    """
    Phase 3: Construct route graph with terminals and lanes.
    
    Returns:
        pd.DataFrame: routes_with_terminals_df
    """
    logging.info("=" * 60)
    logging.info("PHASE 3: ROUTE GRAPH CONSTRUCTION")
    logging.info("=" * 60)
    
    # Step 1: Calculate route centroids
    logging.info("Step 1: Calculating route centroids...")
    route_centroids = calculate_route_centroids(clustered_journeys)
    
    # Step 2: Link routes to terminals
    logging.info("Step 2: Linking routes to terminals...")
    routes_with_terminals = link_routes_to_terminals(
        route_centroids, terminal_summary, clustered_journeys
    )
    
    # Step 3: Save route graph to GeoPackage
    logging.info("Step 3: Saving route graph...")
    save_route_graph_to_geopackage(routes_with_terminals, config['graph']['output_path'])
    
    logging.info(f"Phase 3 complete: {len(routes_with_terminals)} shipping lanes")
    return routes_with_terminals

def phase4_validation_and_visualization(distance_matrix, cluster_labels, terminal_summary, 
                                      clustered_endpoints, routes_with_terminals, config: dict):
    """
    Phase 4: Validate results and create visualization.
    """
    logging.info("=" * 60)
    logging.info("PHASE 4: VALIDATION & VISUALIZATION")
    logging.info("=" * 60)
    
    # Step 1: Calculate validation metrics
    logging.info("Step 1: Calculating validation metrics...")
    clustering_metrics = calculate_clustering_metrics(distance_matrix, cluster_labels)
    terminal_metrics = calculate_terminal_metrics(terminal_summary, clustered_endpoints)
    route_metrics = calculate_route_metrics(routes_with_terminals)
    
    # Step 2: Log validation summary
    logging.info("Step 2: Logging validation summary...")
    log_validation_summary(clustering_metrics, terminal_metrics, route_metrics)
    
    # Step 3: Create and save visualization
    logging.info("Step 3: Creating interactive visualization...")
    create_and_save_shipping_lanes_map(config)
    
    logging.info("Phase 4 complete: Validation and visualization finished")

def main():
    """Main function to run the lane discovery pipeline."""
    parser = argparse.ArgumentParser(description="Discover shipping lanes from AIS data.")
    parser.add_argument('--config', type=str, required=True, 
                       help="Path to the configuration YAML file.")
    args = parser.parse_args()

    logging.info(f"Starting shipping lane discovery with config: {args.config}")

    # Load configuration
    config = load_config(args.config)

    try:
        # Phase 1: Trajectory & Terminal Extraction
        journeys_df, terminal_summary, clustered_endpoints = phase1_trajectory_and_terminal_extraction(config)

        # Phase 2: Route Clustering
        clustered_journeys, distance_matrix, cluster_labels = phase2_route_clustering(journeys_df, config)

        # Phase 3: Route Graph Construction
        routes_with_terminals = phase3_route_graph_construction(clustered_journeys, terminal_summary, config)

        # Phase 4: Validation & Visualization
        phase4_validation_and_visualization(
            distance_matrix, cluster_labels, terminal_summary, 
            clustered_endpoints, routes_with_terminals, config
        )

        logging.info("🎉 Shipping lane discovery pipeline completed successfully!")
        logging.info(f"📊 Results saved to:")
        logging.info(f"   • Terminals: {config['terminals']['output_path']}")
        logging.info(f"   • Routes: {config['graph']['output_path']}")
        logging.info(f"   • Map: {config['visualization']['map_output_path']}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    main()
