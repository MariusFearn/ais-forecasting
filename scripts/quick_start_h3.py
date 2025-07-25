#!/usr/bin/env python3
"""
Quick start script for Phase 1 vessel H3 exploration

This script helps you quickly begin the vessel-level H3 feature engineering
by loading data and running basic analysis.
"""

import pandas as pd
import pickle
import h3
import os
import sys
from pathlib import Path


def find_data_files(data_dir='raw_data'):
    """Find available pickle files"""
    data_path = Path(__file__).parent.parent / data_dir
    
    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        return []
    
    pickle_files = list(data_path.glob('*.pkl'))
    return sorted(pickle_files)


def load_sample_data(file_path, sample_size=10000):
    """Load and sample data for quick exploration"""
    print(f"Loading data from: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        
        print(f"Loaded {len(df):,} records")
        
        # Sample data if too large
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size:,} records for exploration")
            return df_sample
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def quick_vessel_analysis(df):
    """Quick analysis of vessel data"""
    print("\n=== QUICK VESSEL ANALYSIS ===")
    
    # Basic stats
    print(f"Total records: {len(df):,}")
    print(f"Unique vessels: {df['imo'].nunique():,}")
    print(f"Date range: {df['mdt'].min()} to {df['mdt'].max()}")
    
    # Top vessels by record count
    vessel_counts = df['imo'].value_counts()
    print(f"\nTop 5 vessels by record count:")
    for i, (vessel, count) in enumerate(vessel_counts.head().items()):
        print(f"  {i+1}. IMO {vessel}: {count:,} records")
    
    return vessel_counts.head().index.tolist()


def basic_h3_test(df, vessel_imo, h3_resolution=5):
    """Basic H3 testing without the full class"""
    print(f"\n=== BASIC H3 TEST ON VESSEL {vessel_imo} ===")
    
    # Get vessel data
    vessel_data = df[df['imo'] == vessel_imo].sort_values('mdt').copy()
    print(f"Vessel records: {len(vessel_data)}")
    
    # Test H3 conversion
    print(f"Testing H3 resolution {h3_resolution} (edge length: {h3.edge_length(h3_resolution, unit='km'):.2f} km)")
    
    h3_cells = []
    errors = 0
    
    for _, row in vessel_data.iterrows():
        try:
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                if abs(row['lat']) <= 90 and abs(row['lon']) <= 180:
                    cell = h3.geo_to_h3(row['lat'], row['lon'], h3_resolution)
                    h3_cells.append(cell)
                else:
                    h3_cells.append(None)
                    errors += 1
            else:
                h3_cells.append(None)
                errors += 1
        except:
            h3_cells.append(None)
            errors += 1
    
    vessel_data['h3_cell'] = h3_cells
    
    # Basic analysis
    unique_cells = vessel_data['h3_cell'].nunique()
    valid_cells = vessel_data['h3_cell'].notna().sum()
    
    # Cell transitions
    vessel_data['h3_prev'] = vessel_data['h3_cell'].shift(1)
    transitions = (vessel_data['h3_cell'] != vessel_data['h3_prev']).sum()
    
    print(f"✓ H3 conversion complete")
    print(f"  Valid H3 cells: {valid_cells:,} ({valid_cells/len(vessel_data)*100:.1f}%)")
    print(f"  Conversion errors: {errors}")
    print(f"  Unique cells visited: {unique_cells}")
    print(f"  Cell transitions: {transitions}")
    print(f"  Records per cell: {len(vessel_data) / unique_cells:.1f}")
    
    # Time analysis
    vessel_data['time_diff'] = vessel_data['mdt'].diff().dt.total_seconds() / 3600
    print(f"  Median time gap: {vessel_data['time_diff'].median():.2f} hours")
    
    # Speed analysis if available
    if 'speed' in vessel_data.columns:
        print(f"  Average speed: {vessel_data['speed'].mean():.1f} knots")
        print(f"  Stationary time (speed < 1): {(vessel_data['speed'] < 1).mean()*100:.1f}%")
    
    return vessel_data


def main():
    """Main execution function"""
    print("=== VESSEL H3 FEATURE ENGINEERING - QUICK START ===")
    print(f"H3 library version: {h3.__version__}")
    
    # Find data files
    data_files = find_data_files()
    
    if not data_files:
        print("No pickle files found in raw_data/ directory")
        print("Please ensure your AIS data files are in the raw_data/ folder")
        return
    
    print(f"Found {len(data_files)} data files:")
    for i, file_path in enumerate(data_files):
        print(f"  {i+1}. {file_path.name}")
    
    # Load most recent file (or user can modify this)
    latest_file = data_files[-1]  # Assuming files are sorted chronologically
    df = load_sample_data(latest_file, sample_size=50000)  # Larger sample for testing
    
    if df is None:
        return
    
    # Quick analysis
    top_vessels = quick_vessel_analysis(df)
    
    # Test H3 tracking on top vessel
    if top_vessels:
        test_vessel = top_vessels[0]
        h3_data = basic_h3_test(df, test_vessel)
        
        if h3_data is not None:
            print(f"\n✓ Basic H3 test successful!")
            print(f"✓ Ready to continue with full implementation")
            print(f"\nNext steps:")
            print(f"  1. Open the vessel_exploration.ipynb notebook")
            print(f"  2. Run the full analysis on your chosen dataset")
            print(f"  3. Use the VesselH3Tracker class for complete feature engineering")
            print(f"  4. Continue with Phase 2 from todo_create_features.md")
            
            # Test different resolutions
            print(f"\n=== TESTING DIFFERENT H3 RESOLUTIONS ===")
            for res in [4, 5, 6]:
                try:
                    sample_coords = df[['lat', 'lon']].dropna().iloc[0]
                    test_cell = h3.geo_to_h3(sample_coords['lat'], sample_coords['lon'], res)
                    edge_length = h3.edge_length(res, unit='km')
                    print(f"  Resolution {res}: {edge_length:.2f} km edge length")
                except Exception as e:
                    print(f"  Resolution {res}: Error - {e}")


if __name__ == "__main__":
    main()
