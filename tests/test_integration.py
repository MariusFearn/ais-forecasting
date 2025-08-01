#!/usr/bin/env python3
"""
Test Updated Trajectory Processor

Quick test to verify the updated trajectory processor works correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import time

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_updated_trajectory_processor():
    """Test the updated trajectory processor function."""
    print("🔍 Testing Updated Trajectory Processor")
    print("="*50)
    
    try:
        # Import the updated function
        from src.features.trajectory_processor import extract_vessel_trajectories
        print("✅ Function imported successfully")
        
        # Load test data
        from src.data.loader import AISDataLoader
        
        loader = AISDataLoader(data_dir=str(project_root / 'data'), use_duckdb=False)
        print("🔄 Loading test data...")
        
        ais_data = loader.load_multi_year_data_original(["2024"])
        
        # Limit to 3 vessels for quick testing
        unique_vessels = ais_data['imo'].unique()
        test_vessels = unique_vessels[:3]
        test_data = ais_data[ais_data['imo'].isin(test_vessels)].copy()
        
        print(f"📊 Test data: {len(test_data):,} records from {len(test_vessels)} vessels")
        
        # Test with simple config
        config = {
            'h3_resolution': 5,
            'min_journey_length': 5,
            'speed_threshold_knots': 0.5,
            'time_gap_hours': 6
        }
        
        print(f"🔧 Testing with config: {config}")
        start_time = time.time()
        
        trajectories = extract_vessel_trajectories(test_data, config)
        
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.2f} seconds")
        print(f"📊 Results:")
        print(f"   Trajectories extracted: {len(trajectories)}")
        
        if len(trajectories) > 0:
            print(f"   Columns: {list(trajectories.columns)}")
            print(f"   Avg duration: {trajectories['duration_hours'].mean():.1f} hours")
            print(f"   Avg points: {trajectories['point_count'].mean():.1f}")
            print(f"   Avg H3 cells: {trajectories['unique_h3_cells'].mean():.1f}")
            
            # Show sample trajectory
            sample = trajectories.iloc[0]
            print(f"\n📋 Sample trajectory:")
            print(f"   Vessel: {sample['vessel_id']}")
            print(f"   Duration: {sample['duration_hours']:.1f} hours")
            print(f"   Points: {sample['point_count']}")
            print(f"   H3 sequence length: {len(sample['h3_sequence'])}")
            print(f"   Start: ({sample['start_lat']:.4f}, {sample['start_lon']:.4f})")
            print(f"   End: ({sample['end_lat']:.4f}, {sample['end_lon']:.4f})")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("🌊 UPDATED TRAJECTORY PROCESSOR TEST")
    print("="*60)
    
    if test_updated_trajectory_processor():
        print("\n✅ Updated trajectory processor working correctly!")
        print("\n🚀 Ready for pipeline integration!")
        return 0
    else:
        print("\n❌ Updated trajectory processor test failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
