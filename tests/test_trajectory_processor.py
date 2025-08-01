#!/usr/bin/env python3
"""
Minimal Trajectory Processor Test

Creates a simple trajectory processing function and tests it
with real AIS data. No complex dependencies - just basic functionality.
"""

import sys
from pathlib import Path
import pandas as pd
import time
import h3

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def extract_vessel_trajectories_simple(ais_data, config=None):
    """
    Simple trajectory extraction function.
    
    Args:
        ais_data: DataFrame with columns [imo, lat, lon, mdt, speed]
        config: Dictionary with processing parameters
        
    Returns:
        DataFrame with trajectory information
    """
    if config is None:
        config = {
            'h3_resolution': 5,
            'min_journey_length': 3,
            'speed_threshold_knots': 0.5,
            'time_gap_hours': 6
        }
    
    print(f"🔄 Processing {len(ais_data):,} records with config: {config}")
    
    trajectories = []
    
    # Process each vessel separately
    for vessel_id in ais_data['imo'].unique():
        vessel_data = ais_data[ais_data['imo'] == vessel_id].copy()
        vessel_data = vessel_data.sort_values('mdt').reset_index(drop=True)
        
        if len(vessel_data) < config['min_journey_length']:
            continue
            
        # Add H3 cells
        vessel_data['h3_cell'] = vessel_data.apply(
            lambda row: h3.geo_to_h3(row['lat'], row['lon'], config['h3_resolution']), 
            axis=1
        )
        
        # Split into trajectory segments based on time gaps
        vessel_data['time_diff'] = vessel_data['mdt'].diff().dt.total_seconds() / 3600  # hours
        gap_indices = vessel_data[vessel_data['time_diff'] > config['time_gap_hours']].index
        
        # Create trajectory segments
        start_idx = 0
        for gap_idx in list(gap_indices) + [len(vessel_data)]:
            segment = vessel_data.iloc[start_idx:gap_idx].copy()
            
            if len(segment) >= config['min_journey_length']:
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
                    'stationary_points': len(segment[segment['speed'] < config['speed_threshold_knots']]),
                    'h3_sequence': segment['h3_cell'].tolist(),
                    'start_lat': segment['lat'].iloc[0],
                    'start_lon': segment['lon'].iloc[0],
                    'end_lat': segment['lat'].iloc[-1],
                    'end_lon': segment['lon'].iloc[-1]
                }
                trajectories.append(trajectory)
            
            start_idx = gap_idx
    
    print(f"✅ Extracted {len(trajectories)} trajectory segments")
    return pd.DataFrame(trajectories)

def test_simple_trajectory_extraction():
    """Test the simple trajectory extraction function."""
    print("🔍 Testing Simple Trajectory Extraction")
    print("="*50)
    
    try:
        # Load test data
        from src.data.loader import AISDataLoader
        
        loader = AISDataLoader(data_dir=str(project_root / 'data'), use_duckdb=False)
        print("🔄 Loading test data...")
        
        # Load small sample
        ais_data = loader.load_multi_year_data_original(["2024"])
        
        # Limit to 5 vessels for testing
        unique_vessels = ais_data['imo'].unique()
        test_vessels = unique_vessels[:5]
        test_data = ais_data[ais_data['imo'].isin(test_vessels)].copy()
        
        print(f"📊 Test data: {len(test_data):,} records from {len(test_vessels)} vessels")
        
        # Test with different configurations
        configs = [
            {
                'name': 'Relaxed',
                'config': {
                    'h3_resolution': 5,
                    'min_journey_length': 3,
                    'speed_threshold_knots': 0.0,
                    'time_gap_hours': 48
                }
            },
            {
                'name': 'Standard',
                'config': {
                    'h3_resolution': 5,
                    'min_journey_length': 5,
                    'speed_threshold_knots': 0.5,
                    'time_gap_hours': 6
                }
            },
            {
                'name': 'Strict',
                'config': {
                    'h3_resolution': 6,
                    'min_journey_length': 10,
                    'speed_threshold_knots': 1.0,
                    'time_gap_hours': 3
                }
            }
        ]
        
        results = []
        
        for test_config in configs:
            print(f"\n🔧 Testing {test_config['name']} configuration...")
            start_time = time.time()
            
            trajectories = extract_vessel_trajectories_simple(test_data, test_config['config'])
            
            processing_time = time.time() - start_time
            
            if len(trajectories) > 0:
                avg_duration = trajectories['duration_hours'].mean()
                avg_points = trajectories['point_count'].mean()
                avg_cells = trajectories['unique_h3_cells'].mean()
                
                print(f"✅ {test_config['name']} results:")
                print(f"   Trajectories: {len(trajectories)}")
                print(f"   Avg duration: {avg_duration:.1f} hours")
                print(f"   Avg points: {avg_points:.1f}")
                print(f"   Avg H3 cells: {avg_cells:.1f}")
                print(f"   Processing time: {processing_time:.2f}s")
                
                # Sample trajectory analysis
                sample_traj = trajectories.iloc[0]
                print(f"   Sample trajectory:")
                print(f"     Vessel: {sample_traj['vessel_id']}")
                print(f"     Duration: {sample_traj['duration_hours']:.1f}h")
                print(f"     Points: {sample_traj['point_count']}")
                print(f"     H3 cells: {sample_traj['unique_h3_cells']}")
                print(f"     Avg speed: {sample_traj['avg_speed']:.1f} knots")
                
                results.append({
                    'config': test_config['name'],
                    'trajectories': len(trajectories),
                    'processing_time': processing_time
                })
            else:
                print(f"❌ {test_config['name']}: No trajectories extracted")
                results.append({
                    'config': test_config['name'],
                    'trajectories': 0,
                    'processing_time': processing_time
                })
        
        # Summary
        print("\n" + "="*50)
        print("🎯 TRAJECTORY EXTRACTION TEST SUMMARY")
        print("="*50)
        for result in results:
            print(f"{result['config']:>10}: {result['trajectories']:>3} trajectories in {result['processing_time']:.2f}s")
        
        # Recommend best configuration
        best_config = max(results, key=lambda x: x['trajectories'])
        print(f"\n✅ Best configuration: {best_config['config']} ({best_config['trajectories']} trajectories)")
        
        return True
        
    except Exception as e:
        print(f"❌ Trajectory extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the trajectory extraction test."""
    print("🌊 SIMPLE TRAJECTORY PROCESSOR TEST")
    print("="*60)
    
    start_time = time.time()
    
    if test_simple_trajectory_extraction():
        total_time = time.time() - start_time
        print(f"\n✅ All tests passed in {total_time:.1f} seconds!")
        print("\n🚀 Next steps:")
        print("   1. Integrate this function into src/features/trajectory_processor.py")
        print("   2. Test terminal discovery")
        print("   3. Test route clustering")
        print("   4. Run full pipeline")
        return 0
    else:
        print("\n❌ Tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
