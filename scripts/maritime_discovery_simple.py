#!/usr/bin/env python3
"""
Maritime Discovery Pipeline - Simple Working Version

Uses only the validated components that are working correctly.
No complex dependencies - just the basic functionality we've tested.
"""

import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
import time

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Simple maritime discovery pipeline."""
    parser = argparse.ArgumentParser(description='Simple Maritime Discovery Pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--max-vessels', type=int, default=10,
                       help='Maximum number of vessels to process')
    parser.add_argument('--output-dir', type=str,
                       default='./data/processed/test_discovery',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🌊 Simple Maritime Discovery Pipeline")
    print("="*50)
    print(f"Config: {args.config}")
    print(f"Max vessels: {args.max_vessels}")
    print(f"Output: {args.output_dir}")
    
    start_time = time.time()
    
    try:
        # Load configuration
        config = load_config(args.config)
        print("✅ Configuration loaded")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 1: Load AIS Data
        print("\n🔄 Phase 1: Loading AIS data...")
        from src.data.loader import AISDataLoader
        
        loader = AISDataLoader(data_dir=str(project_root / 'data'), use_duckdb=False)
        ais_data = loader.load_multi_year_data_original(["2024"])
        
        # Limit vessels
        unique_vessels = ais_data['imo'].unique()
        if len(unique_vessels) > args.max_vessels:
            selected_vessels = unique_vessels[:args.max_vessels]
            ais_data = ais_data[ais_data['imo'].isin(selected_vessels)]
            
        print(f"✅ Loaded {len(ais_data):,} records from {ais_data['imo'].nunique():,} vessels")
        
        # Phase 2: Extract Trajectories
        print("\n🔄 Phase 2: Extracting trajectories...")
        from src.features.trajectory_processor import extract_vessel_trajectories
        
        trajectory_config = config.get('trajectory_extraction', {})
        trajectories = extract_vessel_trajectories(ais_data, trajectory_config)
        
        if trajectories.empty:
            print("❌ No trajectories extracted")
            return 1
            
        print(f"✅ Extracted {len(trajectories)} trajectory segments")
        
        # Phase 3: Basic Terminal Discovery (simplified)
        print("\n🔄 Phase 3: Discovering terminals...")
        
        # Find stationary periods as potential terminals
        stationary_threshold = config.get('terminal_discovery', {}).get('stationary_speed_threshold', 1.0)
        min_duration = config.get('terminal_discovery', {}).get('min_stationary_hours', 1.0)
        
        import h3
        
        terminals = []
        for _, vessel_data in ais_data.groupby('imo'):
            vessel_data = vessel_data.sort_values('mdt')
            
            # Find stationary periods
            stationary = vessel_data[vessel_data['speed'] < stationary_threshold].copy()
            
            if len(stationary) > 0:
                # Group by H3 cells
                stationary['h3_cell'] = stationary.apply(
                    lambda row: h3.geo_to_h3(row['lat'], row['lon'], 6), axis=1
                )
                
                for h3_cell, cell_data in stationary.groupby('h3_cell'):
                    if len(cell_data) >= 3:  # Minimum points
                        duration = (cell_data['mdt'].max() - cell_data['mdt'].min()).total_seconds() / 3600
                        
                        if duration >= min_duration:
                            terminal = {
                                'h3_cell': h3_cell,
                                'lat': cell_data['lat'].mean(),
                                'lon': cell_data['lon'].mean(),
                                'vessel_count': 1,
                                'total_visits': len(cell_data),
                                'duration_hours': duration,
                                'vessels': [vessel_data['imo'].iloc[0]]
                            }
                            terminals.append(terminal)
        
        terminals_df = pd.DataFrame(terminals)
        
        # Consolidate terminals by location
        if not terminals_df.empty:
            terminal_summary = terminals_df.groupby('h3_cell').agg({
                'lat': 'mean',
                'lon': 'mean',
                'vessel_count': 'sum',
                'total_visits': 'sum',
                'duration_hours': 'sum'
            }).reset_index()
            
            print(f"✅ Discovered {len(terminal_summary)} potential terminals")
        else:
            terminal_summary = pd.DataFrame()
            print("⚠️ No terminals discovered")
        
        # Phase 4: Save Results
        print("\n🔄 Phase 4: Saving results...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trajectories
        trajectory_file = output_dir / f"trajectories_{timestamp}.parquet"
        trajectories.to_parquet(trajectory_file, index=False)
        print(f"📁 Trajectories saved: {trajectory_file}")
        
        # Save terminals
        if not terminal_summary.empty:
            terminals_file = output_dir / f"terminals_{timestamp}.parquet"
            terminal_summary.to_parquet(terminals_file, index=False)
            print(f"📁 Terminals saved: {terminals_file}")
        
        # Create summary
        summary = {
            'pipeline_run': {
                'timestamp': datetime.now().isoformat(),
                'config_file': args.config,
                'max_vessels': args.max_vessels,
                'processing_time_seconds': time.time() - start_time
            },
            'data_summary': {
                'total_ais_records': len(ais_data),
                'unique_vessels': ais_data['imo'].nunique(),
                'trajectories_extracted': len(trajectories),
                'terminals_discovered': len(terminal_summary) if not terminal_summary.empty else 0
            },
            'trajectory_analysis': {
                'avg_duration_hours': trajectories['duration_hours'].mean() if not trajectories.empty else 0,
                'avg_points_per_trajectory': trajectories['point_count'].mean() if not trajectories.empty else 0,
                'avg_h3_cells_per_trajectory': trajectories['unique_h3_cells'].mean() if not trajectories.empty else 0
            },
            'output_files': {
                'trajectories': str(trajectory_file),
                'terminals': str(terminals_file) if not terminal_summary.empty else None
            }
        }
        
        summary_file = output_dir / f"summary_{timestamp}.yaml"
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        # Final summary
        total_time = time.time() - start_time
        
        print("\n" + "="*50)
        print("🎯 PIPELINE SUMMARY")
        print("="*50)
        print(f"📊 AIS Records: {len(ais_data):,}")
        print(f"🚢 Vessels: {ais_data['imo'].nunique():,}")
        print(f"🛣️ Trajectories: {len(trajectories):,}")
        print(f"🏢 Terminals: {len(terminal_summary) if not terminal_summary.empty else 0}")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        print(f"📁 Results: {output_dir}")
        print("="*50)
        
        print("\n✅ Simple maritime discovery pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
