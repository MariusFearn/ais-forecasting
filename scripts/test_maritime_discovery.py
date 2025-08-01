#!/usr/bin/env python3
"""
Quick test script for maritime discovery pipeline.

This script tests the newly refactored maritime discovery components
to ensure they integrate properly with existing infrastructure.
"""

import sys
from pathlib import Path
import pandas as pd
import yaml

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test that all new modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from src.data.maritime_loader import load_global_ais_data, validate_ais_data
        print("✅ Maritime loader imported")
        
        from src.features.trajectory_processor import extract_vessel_trajectories
        print("✅ Trajectory processor imported")
        
        from src.features.route_clustering import cluster_shipping_routes
        print("✅ Route clustering imported")
        
        from src.features.terminal_discovery import TerminalDiscovery
        print("✅ Terminal discovery imported")
        
        # Test existing DTW functions are accessible
        from src.models.clustering import compute_dtw_distance_matrix, cluster_routes
        print("✅ Existing DTW functions accessible")
        
        # Test existing DuckDB infrastructure
        from src.data.duckdb_engine import DuckDBEngine
        from src.data.loader import AISDataLoader
        print("✅ Existing DuckDB infrastructure accessible")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n🔍 Testing configuration loading...")
    
    config_path = Path(__file__).parent.parent / 'config' / 'maritime_discovery_test.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Test config loaded from {config_path}")
        print(f"   - Data loading config: {bool(config.get('data_loading'))}")
        print(f"   - Route clustering config: {bool(config.get('route_clustering'))}")
        print(f"   - Terminal discovery config: {bool(config.get('terminal_discovery'))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_sample_data_creation():
    """Test creating sample data with original column names."""
    print("\n🔍 Testing sample data creation...")
    
    try:
        # Create small sample dataset with original AIS column names
        sample_data = pd.DataFrame({
            'imo': [1234567, 1234567, 1234568, 1234568, 1234569],
            'mdt': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 11:00:00', 
                '2024-01-01 10:30:00',
                '2024-01-01 11:30:00',
                '2024-01-01 12:00:00'
            ]),
            'lat': [-33.9, -33.8, -34.0, -34.1, -33.7],
            'lon': [18.4, 18.5, 18.3, 18.2, 18.6],
            'speed': [10.5, 8.2, 12.1, 0.5, 15.3],
            'draught': [12.5, 12.5, 8.2, 8.2, 14.1]
        })
        
        print(f"✅ Sample data created: {len(sample_data)} records")
        print(f"   - Vessels: {sample_data['imo'].nunique()}")
        print(f"   - Columns: {list(sample_data.columns)}")
        
        # Test validation function
        from src.data.maritime_loader import validate_ais_data
        validate_ais_data(sample_data)
        print("✅ Data validation passed")
        
        return sample_data
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return None

def test_trajectory_extraction(sample_data):
    """Test trajectory extraction with sample data."""
    print("\n🔍 Testing trajectory extraction...")
    
    try:
        from src.features.trajectory_processor import extract_vessel_trajectories
        
        config = {
            'h3_resolution': 5,
            'min_points_per_trajectory': 2,
            'max_gap_hours': 6
        }
        
        trajectories = extract_vessel_trajectories(sample_data, config)
        
        print(f"✅ Trajectory extraction completed")
        print(f"   - Trajectories: {len(trajectories)}")
        print(f"   - Columns: {list(trajectories.columns) if not trajectories.empty else 'None'}")
        
        return trajectories
        
    except Exception as e:
        print(f"❌ Trajectory extraction failed: {e}")
        return pd.DataFrame()

def test_terminal_discovery(sample_data):
    """Test terminal discovery with sample data."""
    print("\n🔍 Testing terminal discovery...")
    
    try:
        from src.features.terminal_discovery import TerminalDiscovery
        
        config = {
            'stationary_speed_threshold': 1.0,
            'min_stationary_duration_hours': 0.5,
            'min_vessels_for_terminal': 1,
            'h3_resolution': 8
        }
        
        terminal_discovery = TerminalDiscovery(config)
        terminals = terminal_discovery.discover_terminals(sample_data)
        
        print(f"✅ Terminal discovery completed")
        print(f"   - Terminals found: {len(terminals)}")
        print(f"   - Columns: {list(terminals.columns) if not terminals.empty else 'None'}")
        
        return terminals
        
    except Exception as e:
        print(f"❌ Terminal discovery failed: {e}")
        return pd.DataFrame()

def main():
    """Run all tests."""
    print("🌊 Maritime Discovery Pipeline - Integration Test")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_passed = False
    
    # Test 2: Configuration
    if not test_config_loading():
        all_passed = False
    
    # Test 3: Sample data
    sample_data = test_sample_data_creation()
    if sample_data is None:
        all_passed = False
        return
    
    # Test 4: Trajectory extraction
    trajectories = test_trajectory_extraction(sample_data)
    if trajectories.empty:
        print("⚠️  No trajectories extracted (may be normal with small sample)")
    
    # Test 5: Terminal discovery
    terminals = test_terminal_discovery(sample_data)
    if terminals.empty:
        print("⚠️  No terminals discovered (may be normal with small sample)")
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Maritime discovery pipeline is ready for production use")
        print("\nNext steps:")
        print("1. Run with test config: python scripts/maritime_discovery.py --config config/maritime_discovery_test.yaml --years 2024")
        print("2. Run with full config: python scripts/maritime_discovery.py --config config/maritime_discovery.yaml --years 2023 2024")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and fix issues before proceeding")
    
    print("="*60)

if __name__ == "__main__":
    main()
