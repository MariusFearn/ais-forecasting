#!/usr/bin/env python3
"""
Maritime Discovery Pipeline - Fast Testing

Simple, fast testing script without notebook overhead.
Tests core functionality step by step with immediate feedback.
"""

import sys
import time
from pathlib import Path
import pandas as pd

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_basic_imports():
    """Test basic imports that should work."""
    print("🔍 Test 1: Basic imports...")
    
    try:
        from src.data.loader import AISDataLoader
        print("✅ AISDataLoader imported")
        
        # Test data loading
        loader = AISDataLoader(data_dir=str(project_root / 'data'), use_duckdb=False)
        print("✅ AISDataLoader initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_data_loading():
    """Test actual data loading with small sample."""
    print("\n🔍 Test 2: Data loading...")
    
    try:
        from src.data.loader import AISDataLoader
        
        # Load small sample
        loader = AISDataLoader(data_dir=str(project_root / 'data'), use_duckdb=False)
        
        print("🔄 Loading 2024 data...")
        start_time = time.time()
        ais_data = loader.load_multi_year_data_original(["2024"])
        load_time = time.time() - start_time
        
        if ais_data.empty:
            print("❌ No data loaded")
            return False
            
        print(f"✅ Data loaded in {load_time:.1f}s:")
        print(f"   Records: {len(ais_data):,}")
        print(f"   Vessels: {ais_data['imo'].nunique():,}")
        print(f"   Columns: {list(ais_data.columns)}")
        
        return ais_data
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_analysis(ais_data):
    """Test basic data analysis."""
    print("\n🔍 Test 3: Basic analysis...")
    
    try:
        # Limit to small sample for testing
        unique_vessels = ais_data['imo'].unique()
        test_vessels = unique_vessels[:3]
        test_data = ais_data[ais_data['imo'].isin(test_vessels)].copy()
        
        print(f"📊 Test sample: {len(test_data):,} records from {len(test_vessels)} vessels")
        
        # Basic analysis
        for vessel_id in test_vessels:
            vessel_data = test_data[test_data['imo'] == vessel_id]
            vessel_data = vessel_data.sort_values('mdt')
            
            duration = (vessel_data['mdt'].max() - vessel_data['mdt'].min()).days
            avg_speed = vessel_data['speed'].mean()
            stationary = len(vessel_data[vessel_data['speed'] < 1.0])
            
            print(f"🚢 Vessel {vessel_id}:")
            print(f"   Records: {len(vessel_data):,}")
            print(f"   Duration: {duration} days")
            print(f"   Avg speed: {avg_speed:.1f} knots")
            print(f"   Stationary points: {stationary} ({stationary/len(vessel_data)*100:.1f}%)")
            
        print("✅ Basic analysis complete")
        return test_data
        
    except Exception as e:
        print(f"❌ Basic analysis failed: {e}")
        return False

def test_h3_functionality():
    """Test H3 spatial indexing."""
    print("\n🔍 Test 4: H3 spatial indexing...")
    
    try:
        import h3
        
        # Test H3 with Cape Town coordinates
        cape_town = (-33.9249, 18.4241)
        h3_cell = h3.geo_to_h3(cape_town[0], cape_town[1], 5)
        
        print(f"📍 Cape Town: {cape_town}")
        print(f"   H3 cell (res 5): {h3_cell}")
        
        # Test multiple resolutions
        for res in [3, 5, 7]:
            cell = h3.geo_to_h3(cape_town[0], cape_town[1], res)
            print(f"   Resolution {res}: {cell}")
            
        print("✅ H3 functionality working")
        return True
        
    except Exception as e:
        print(f"❌ H3 test failed: {e}")
        return False

def test_trajectory_processing_concept(test_data):
    """Test basic trajectory processing concepts."""
    print("\n🔍 Test 5: Trajectory processing concepts...")
    
    try:
        import h3
        
        # Process one vessel to demonstrate trajectory concepts
        vessel_id = test_data['imo'].iloc[0]
        vessel_data = test_data[test_data['imo'] == vessel_id].sort_values('mdt')
        
        print(f"🚢 Processing vessel {vessel_id} ({len(vessel_data)} points)")
        
        # Add H3 cells
        vessel_data['h3_cell'] = vessel_data.apply(
            lambda row: h3.geo_to_h3(row['lat'], row['lon'], 5), axis=1
        )
        
        # Basic trajectory metrics
        unique_cells = vessel_data['h3_cell'].nunique()
        time_span = (vessel_data['mdt'].max() - vessel_data['mdt'].min()).total_seconds() / 3600
        avg_speed = vessel_data['speed'].mean()
        
        print(f"   Unique H3 cells: {unique_cells}")
        print(f"   Time span: {time_span:.1f} hours")
        print(f"   Average speed: {avg_speed:.1f} knots")
        
        # Identify potential terminals (stationary periods)
        stationary = vessel_data[vessel_data['speed'] < 1.0]
        if len(stationary) > 0:
            stationary_cells = stationary['h3_cell'].value_counts()
            print(f"   Potential terminals: {len(stationary_cells)} H3 cells")
            if len(stationary_cells) > 0:
                top_terminal = stationary_cells.index[0]
                print(f"   Top terminal cell: {top_terminal} ({stationary_cells.iloc[0]} points)")
        
        print("✅ Trajectory processing concepts working")
        return True
        
    except Exception as e:
        print(f"❌ Trajectory processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_config():
    """Create a simple test configuration."""
    print("\n🔍 Creating simple test configuration...")
    
    import yaml
    
    config = {
        'data_loading': {
            'use_duckdb': False,
            'max_vessels': 10
        },
        'trajectory_extraction': {
            'h3_resolution': 5,
            'min_journey_length': 3,
            'speed_threshold_knots': 0.5,
            'time_gap_hours': 6
        },
        'terminal_discovery': {
            'stationary_speed_threshold': 1.0,
            'min_stationary_hours': 1.0,
            'min_vessels_for_terminal': 2
        }
    }
    
    # Save config
    config_dir = project_root / 'config'
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / 'maritime_test_simple.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"✅ Simple config created: {config_file}")
    return config_file

def main():
    """Run all tests quickly."""
    print("🌊 MARITIME DISCOVERY - FAST TESTING")
    print("="*50)
    
    start_time = time.time()
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("❌ Cannot continue - basic imports failed")
        return 1
        
    # Test 2: Data loading
    ais_data = test_data_loading()
    if ais_data is False:
        print("❌ Cannot continue - data loading failed")
        return 1
        
    # Test 3: Basic analysis
    test_data = test_basic_analysis(ais_data)
    if test_data is False:
        print("❌ Basic analysis failed")
        return 1
        
    # Test 4: H3 functionality
    if not test_h3_functionality():
        print("❌ H3 functionality failed")
        return 1
        
    # Test 5: Trajectory concepts
    if not test_trajectory_processing_concept(test_data):
        print("❌ Trajectory processing failed")
        return 1
        
    # Test 6: Configuration
    config_file = create_simple_config()
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*50)
    print("🎯 FAST TEST SUMMARY")
    print("="*50)
    print(f"✅ All core tests passed!")
    print(f"📊 Data: {len(ais_data):,} records from {ais_data['imo'].nunique():,} vessels")
    print(f"⏱️ Total time: {total_time:.1f} seconds")
    print(f"📁 Config: {config_file}")
    print("="*50)
    print("\n🚀 READY FOR PRODUCTION TESTING!")
    print("💡 Next steps:")
    print("   1. Test individual functions")
    print("   2. Create minimal trajectory processor")
    print("   3. Test small-scale pipeline")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
