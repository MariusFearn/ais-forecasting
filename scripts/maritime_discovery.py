#!/usr/bin/env python3
"""
Maritime Discovery Pipeline - Production Script

Comprehensive maritime traffic analysis using optimized infrastructure:
- Leverages existing DuckDB engine (10-50x speedup)
- Uses original AIS column names (imo, mdt, lat, lon)
- Integrates existing DTW clustering functions
- Implements efficient terminal discovery algorithm

This script transforms the successful notebook exploration into a production pipeline.
"""

import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# Add src to Python path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data.maritime_loader import load_global_ais_data, validate_ais_data
from src.features.trajectory_processor import (
    extract_vessel_trajectories, 
    process_trajectories_batch,
    calculate_trajectory_metrics
)
from src.features.route_clustering import cluster_shipping_routes, analyze_route_clusters
from src.features.terminal_discovery import TerminalDiscovery, extract_terminal_locations
from src.utils.logging_setup import setup_logging, get_logger

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main maritime discovery pipeline."""
    parser = argparse.ArgumentParser(description='Maritime Discovery Pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--years', type=str, nargs='+', 
                       default=['2024'],
                       help='Years to process (e.g., 2023 2024)')
    parser.add_argument('--max-vessels', type=int,
                       help='Maximum number of vessels to process (for testing)')
    parser.add_argument('--output-dir', type=str,
                       default='./data/processed/maritime_discovery',
                       help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"maritime_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file, args.log_level)
    logger = get_logger(__name__)
    
    logger.info("🌊 Starting Maritime Discovery Pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Years: {args.years}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("✅ Configuration loaded successfully")
        
        # Phase 1: Load Global AIS Data (using existing DuckDB optimization)
        logger.info("🔄 Phase 1: Loading global AIS data")
        ais_data = load_global_ais_data(
            years=args.years,
            config=config.get('data_loading', {}),
            max_vessels=args.max_vessels
        )
        
        if ais_data.empty:
            logger.error("No AIS data loaded. Exiting.")
            return 1
        
        # Validate data structure
        validate_ais_data(ais_data)
        logger.info(f"✅ Loaded {len(ais_data):,} AIS records from {ais_data['imo'].nunique():,} vessels")
        
        # Phase 2: Extract Vessel Trajectories
        logger.info("🔄 Phase 2: Extracting vessel trajectories")
        trajectories = extract_vessel_trajectories(
            ais_data,
            config.get('trajectory_extraction', {})
        )
        
        if trajectories.empty:
            logger.error("No trajectories extracted. Exiting.")
            return 1
        
        logger.info(f"✅ Extracted {len(trajectories):,} trajectory segments")
        
        # Phase 3: Process Trajectories in Batches
        logger.info("🔄 Phase 3: Processing trajectories")
        processed_trajectories = process_trajectories_batch(
            trajectories,
            config.get('trajectory_processing', {}),
            batch_size=config.get('batch_size', 1000)
        )
        
        # Calculate trajectory metrics
        trajectory_metrics = calculate_trajectory_metrics(processed_trajectories)
        logger.info(f"✅ Processed trajectories with metrics")
        
        # Phase 4: Route Clustering (using existing DTW functions)
        logger.info("🔄 Phase 4: Clustering shipping routes")
        clustered_trajectories = cluster_shipping_routes(
            processed_trajectories,
            config.get('route_clustering', {}),
            max_routes=config.get('max_routes_for_dtw', 150)
        )
        
        # Analyze clustering results
        clustering_analysis = analyze_route_clusters(clustered_trajectories)
        logger.info(f"✅ Route clustering complete: {clustering_analysis.get('clusters_found', 0)} clusters")
        
        # Phase 5: Terminal Discovery
        logger.info("🔄 Phase 5: Discovering maritime terminals")
        terminal_discovery = TerminalDiscovery(config.get('terminal_discovery', {}))
        terminals = terminal_discovery.discover_terminals(ais_data)
        
        terminal_locations = extract_terminal_locations(terminals)
        logger.info(f"✅ Discovered {len(terminals)} maritime terminals")
        
        # Phase 6: Save Results
        logger.info("🔄 Phase 6: Saving results")
        
        # Save trajectory data
        trajectory_file = output_dir / f"trajectories_{datetime.now().strftime('%Y%m%d')}.parquet"
        clustered_trajectories.to_parquet(trajectory_file)
        logger.info(f"Saved trajectories: {trajectory_file}")
        
        # Save trajectory metrics
        metrics_file = output_dir / f"trajectory_metrics_{datetime.now().strftime('%Y%m%d')}.parquet"
        trajectory_metrics.to_parquet(metrics_file)
        logger.info(f"Saved trajectory metrics: {metrics_file}")
        
        # Save terminals
        if not terminals.empty:
            terminals_file = output_dir / f"terminals_{datetime.now().strftime('%Y%m%d')}.parquet"
            terminals.to_parquet(terminals_file)
            logger.info(f"Saved terminals: {terminals_file}")
        
        # Save clustering analysis
        analysis_file = output_dir / f"clustering_analysis_{datetime.now().strftime('%Y%m%d')}.yaml"
        with open(analysis_file, 'w') as f:
            yaml.dump(clustering_analysis, f, default_flow_style=False)
        logger.info(f"Saved clustering analysis: {analysis_file}")
        
        # Generate summary report
        summary_file = output_dir / f"discovery_summary_{datetime.now().strftime('%Y%m%d')}.yaml"
        summary = {
            'pipeline_run': {
                'timestamp': datetime.now().isoformat(),
                'config_file': args.config,
                'years_processed': args.years,
                'max_vessels': args.max_vessels
            },
            'data_summary': {
                'total_ais_records': len(ais_data),
                'unique_vessels': ais_data['imo'].nunique(),
                'trajectories_extracted': len(trajectories),
                'trajectories_processed': len(processed_trajectories)
            },
            'route_clustering': clustering_analysis,
            'terminal_discovery': {
                'terminals_found': len(terminals),
                'terminal_types': terminals['terminal_type'].value_counts().to_dict() if not terminals.empty and 'terminal_type' in terminals.columns else {}
            },
            'output_files': {
                'trajectories': str(trajectory_file),
                'trajectory_metrics': str(metrics_file),
                'terminals': str(terminals_file) if not terminals.empty else None,
                'clustering_analysis': str(analysis_file)
            }
        }
        
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"✅ Pipeline complete! Summary saved: {summary_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("🌊 MARITIME DISCOVERY PIPELINE SUMMARY")
        print("="*60)
        print(f"📊 AIS Records Processed: {len(ais_data):,}")
        print(f"🚢 Unique Vessels: {ais_data['imo'].nunique():,}")
        print(f"🛣️  Trajectories: {len(processed_trajectories):,}")
        print(f"🔗 Route Clusters: {clustering_analysis.get('clusters_found', 0)}")
        print(f"🏢 Terminals Discovered: {len(terminals)}")
        print(f"📁 Results saved to: {output_dir}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
