#!/usr/bin/env python3
"""
Simple test script to verify our vessel feature extraction works
Tests on just 1 vessel with limited data
"""

import pandas as pd
import sys
sys.path.append('/home/marius/repo_linux/ais-forecasting/src')

from features.vessel_h3_tracker import VesselH3Tracker
from features.vessel_features import VesselFeatureExtractor

def test_single_vessel():
    """Test on just one vessel with limited data"""
    
    print("🚀 Testing Vessel Feature Extraction...")
    
    # Load a small sample of data
    print("\n📊 Loading test data...")
    df = pd.read_pickle('/home/marius/repo_linux/ais-forecasting/data/raw/ais_cape_data_2024.pkl')
    
    # Take just one vessel with decent amount of data
    vessel_counts = df['imo'].value_counts()
    test_vessel = vessel_counts.index[0]  # Vessel with most records
    vessel_data = df[df['imo'] == test_vessel].head(100).copy()  # Just first 100 records
    
    print(f"🚢 Testing vessel {test_vessel}: {len(vessel_data)} records")
    
    # Step 1: Convert to H3 sequence
    print("\n🗺️  Step 1: Converting to H3 sequence...")
    tracker = VesselH3Tracker(h3_resolution=5)
    h3_sequence = tracker.convert_vessel_to_h3_sequence(vessel_data)
    print(f"   ✅ H3 sequence created: {len(h3_sequence)} positions")
    
    # Step 2: Extract features
    print("\n🔧 Step 2: Extracting vessel features...")
    extractor = VesselFeatureExtractor(h3_resolution=5)
    features_df = extractor.extract_all_features(h3_sequence)
    print(f"   ✅ Features extracted: {len(features_df)} rows, {len(features_df.columns)} features")
    
    # Step 3: Show sample results
    print("\n📋 Sample features:")
    print(features_df[['current_h3_cell', 'current_speed', 'current_heading', 'time_in_current_cell']].head())
    
    print(f"\n🎯 SUCCESS! Feature extraction working on {len(features_df)} data points")
    print(f"Feature columns: {list(features_df.columns)[:10]}...")  # First 10 features
    
    return features_df

if __name__ == "__main__":
    try:
        result = test_single_vessel()
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
