#!/usr/bin/env python3
"""
Global Maritime Terminal Discovery Pipeline

Executes the complete global terminal discovery workflow:
1. Load worldwide AIS data (2018-2024)
2. Process vessel trajectories
3. Discover maritime terminals
4. Cluster shipping routes
5. Generate interactive visualizations

Usage:
    python scripts/discover_global_maritime_terminals.py --config config/global_maritime_discovery.yaml
    python scripts/discover_global_maritime_terminals.py --config config/global_maritime_discovery.yaml --experiment experiments/global_maritime_discovery/experiment_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import time
from typing import Dict, Any

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.maritime_loader import load_global_ais_data, get_data_summary
from src.features.trajectory_processor import extract_vessel_trajectories, calculate_trajectory_metrics
from src.utils.config import load_config, merge_experiment_config, validate_config, get_output_paths
from src.utils.logging_setup import setup_logging, get_logger, log_performance_metrics

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Global Maritime Terminal Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default configuration
    python scripts/discover_global_maritime_terminals.py

    # Run with custom configuration
    python scripts/discover_global_maritime_terminals.py --config config/global_maritime_discovery.yaml

    # Run with experiment configuration
    python scripts/discover_global_maritime_terminals.py \\
        --config config/global_maritime_discovery.yaml \\
        --experiment experiments/global_maritime_discovery/experiment_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/global_maritime_discovery.yaml',
        help='Path to main configuration file (default: config/global_maritime_discovery.yaml)'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        help='Path to experiment configuration file (optional)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (optional, logs to console if not specified)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and show pipeline steps without execution'
    )
    
    return parser.parse_args()

def load_and_validate_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and validate configuration files."""
    
    # Load base configuration
    config = load_config(args.config)
    
    # Load experiment configuration if specified
    if args.experiment:
        experiment_config = load_config(args.experiment)
        config = merge_experiment_config(config, experiment_config)
    
    # Override log level if specified
    if args.log_level:
        config['performance']['log_level'] = args.log_level
    
    # Validate configuration
    validate_config(config)
    
    return config

def show_pipeline_summary(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Display pipeline configuration summary."""
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("GLOBAL MARITIME TERMINAL DISCOVERY PIPELINE")
    logger.info("=" * 60)
    
    # Configuration info
    logger.info(f"📋 Configuration: {args.config}")
    if args.experiment:
        logger.info(f"🧪 Experiment: {args.experiment}")
    
    # Data info
    data_files = config['data']['input_files']
    logger.info(f"📁 Input files: {len(data_files)} AIS datasets")
    logger.info(f"📅 Date range: {config['data']['date_range']['start']} to {config['data']['date_range']['end']}")
    
    # Processing parameters
    logger.info(f"🗺️ H3 resolution: {config['processing']['h3_resolution']}")
    logger.info(f"🚢 Journey length: {config['processing']['min_journey_length']}-{config['processing']['max_journey_length']} points")
    logger.info(f"🏴 Terminal criteria: ≥{config['terminals']['min_visits']} visits, ≥{config['terminals']['min_vessels']} vessels")
    
    # Output paths
    output_paths = get_output_paths(config, create_dirs=False)
    logger.info(f"💾 Outputs:")
    for output_type, path in output_paths.items():
        logger.info(f"   {output_type}: {path}")
    
    logger.info("=" * 60)

def execute_phase1_data_loading(config: Dict[str, Any]) -> Any:
    """Execute Phase 1: Data Loading."""
    logger = get_logger(__name__)
    logger.info("🔄 PHASE 1: GLOBAL AIS DATA LOADING")
    
    # Load global AIS data
    ais_data = load_global_ais_data(
        file_paths=config['data']['input_files'],
        date_range=(
            config['data']['date_range']['start'],
            config['data']['date_range']['end']
        ),
        memory_limit_gb=config['processing'].get('memory_limit_gb')
    )
    
    # Generate and log summary
    summary = get_data_summary(ais_data)
    logger.info(f"📊 Data Summary:")
    logger.info(f"   Total records: {summary['total_records']:,}")
    logger.info(f"   Unique vessels: {summary['unique_vessels']:,}")
    logger.info(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    logger.info(f"   Memory usage: {summary['memory_usage_mb']:.1f} MB")
    
    return ais_data

def execute_phase2_trajectory_processing(ais_data: Any, config: Dict[str, Any]) -> Any:
    """Execute Phase 2: Trajectory Processing."""
    logger = get_logger(__name__)
    logger.info("🔄 PHASE 2: VESSEL TRAJECTORY PROCESSING")
    
    # Extract trajectories
    journeys = extract_vessel_trajectories(
        ais_data=ais_data,
        h3_resolution=config['processing']['h3_resolution'],
        min_journey_length=config['processing']['min_journey_length'],
        max_journey_length=config['processing']['max_journey_length'],
        speed_threshold_knots=config['processing'].get('speed_threshold_knots', 0.5),
        time_gap_hours=config['processing'].get('time_gap_hours', 6)
    )
    
    # Calculate and log metrics
    metrics = calculate_trajectory_metrics(journeys)
    logger.info(f"📊 Trajectory Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"   {metric}: {value:.1f}")
        else:
            logger.info(f"   {metric}: {value:,}")
    
    return journeys

def execute_phase3_terminal_discovery(journeys: Any, config: Dict[str, Any]) -> Any:
    """Execute Phase 3: Terminal Discovery."""
    logger = get_logger(__name__)
    logger.info("🔄 PHASE 3: MARITIME TERMINAL DISCOVERY")
    
    # TODO: Implement terminal discovery
    # This will be implemented in the next phase
    logger.info("⚠️ Terminal discovery implementation pending...")
    return None

def execute_phase4_route_clustering(journeys: Any, config: Dict[str, Any]) -> Any:
    """Execute Phase 4: Route Clustering."""
    logger = get_logger(__name__)
    logger.info("🔄 PHASE 4: SHIPPING ROUTE CLUSTERING")
    
    # TODO: Implement route clustering
    # This will be implemented in the next phase
    logger.info("⚠️ Route clustering implementation pending...")
    return None

def execute_phase5_visualization(terminals: Any, routes: Any, config: Dict[str, Any]) -> Any:
    """Execute Phase 5: Interactive Visualization."""
    logger = get_logger(__name__)
    logger.info("🔄 PHASE 5: INTERACTIVE VISUALIZATION")
    
    # TODO: Implement visualization
    # This will be implemented in the next phase
    logger.info("⚠️ Visualization implementation pending...")
    return None

def save_results(
    ais_data: Any,
    journeys: Any, 
    terminals: Any, 
    routes: Any, 
    visualization: Any, 
    config: Dict[str, Any]
) -> None:
    """Save all pipeline results."""
    logger = get_logger(__name__)
    logger.info("💾 Saving pipeline results...")
    
    output_paths = get_output_paths(config, create_dirs=True)
    
    # Save trajectories (always available)
    if journeys is not None and not journeys.empty:
        trajectory_path = Path(config['outputs']['base_directory']) / "global_trajectories.parquet"
        journeys.to_parquet(trajectory_path, index=False)
        logger.info(f"✅ Trajectories saved: {trajectory_path}")
    
    # TODO: Save other outputs when implemented
    logger.info("💾 Results saving complete")

def main() -> int:
    """Execute the complete global maritime discovery pipeline."""
    
    # Parse arguments and load configuration
    args = parse_arguments()
    
    try:
        config = load_and_validate_config(args)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1
    
    # Setup logging
    log_file = args.log_file
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    setup_logging(
        level=config['performance']['log_level'],
        log_file=log_file
    )
    
    logger = get_logger(__name__)
    
    # Show pipeline summary
    show_pipeline_summary(config, args)
    
    # Dry run mode
    if args.dry_run:
        logger.info("🔍 Dry run complete - configuration validated successfully")
        return 0
    
    # Execute pipeline
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting global maritime discovery pipeline...")
        
        # Phase 1: Data Loading
        ais_data = execute_phase1_data_loading(config)
        
        # Phase 2: Trajectory Processing
        journeys = execute_phase2_trajectory_processing(ais_data, config)
        
        # Phase 3: Terminal Discovery
        terminals = execute_phase3_terminal_discovery(journeys, config)
        
        # Phase 4: Route Clustering
        routes = execute_phase4_route_clustering(journeys, config)
        
        # Phase 5: Visualization
        visualization = execute_phase5_visualization(terminals, routes, config)
        
        # Save results
        save_results(ais_data, journeys, terminals, routes, visualization, config)
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️ Total processing time: {total_time/60:.1f} minutes")
        
        # Log final metrics
        log_performance_metrics(
            logger,
            "Complete Pipeline",
            total_time,
            len(journeys) if journeys is not None else 0
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logger.exception("Full error traceback:")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
