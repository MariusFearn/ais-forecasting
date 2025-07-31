#!/usr/bin/env python3
"""
Test script to see what features are actually extracted by VesselFeatureExtractor
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from features.vessel_h3_tracker import VesselH3Tracker
from features.vessel_features import VesselFeatureExtractor

def test_feature_extraction():
    """Test what features are actually extracted"""
    print("🧪 Testing VesselFeatureExtractor...")
    
    # Load sample data from correct path
    print("📊 Loading sample data...")
    df = pd.read_pickle('data/raw/ais_cape_data_2024.pkl')  # Fixed path
    
    # Take a small sample from one vessel
    vessel_counts = df['imo'].value_counts()
    test_vessel = vessel_counts.index[0]
    sample_data = df[df['imo'] == test_vessel].head(100).copy()
    
    print(f"🚢 Using vessel {test_vessel}: {len(sample_data)} records")
    
    # Convert to H3 sequence
    print("🗺️  Converting to H3 sequence...")
    tracker = VesselH3Tracker(h3_resolution=5)
    h3_data = tracker.convert_vessel_to_h3_sequence(sample_data)
    print(f"   H3 sequence: {len(h3_data)} positions")
    
    # Extract features
    print("🔧 Extracting features...")
    extractor = VesselFeatureExtractor(h3_resolution=5)
    features = extractor.extract_all_features(h3_data)
    
    print(f"\n📊 FEATURE EXTRACTION RESULTS:")
    print(f"Total columns: {len(features.columns)}")
    print(f"Total rows: {len(features)}")
    
    print(f"\n📋 All extracted features:")
    for i, col in enumerate(features.columns, 1):
        # Check if feature has meaningful values (not all NaN, not all same value)
        non_nan_count = features[col].notna().sum()
        unique_count = features[col].nunique()
        
        if non_nan_count == 0:
            status = "❌ ALL NaN"
        elif unique_count == 1:
            status = "⚠️  CONSTANT"
        elif unique_count < 3:
            status = "⚠️  LIMITED"
        else:
            status = "✅ GOOD"
            
        print(f"{i:2d}. {col:<30} {status} (non-null: {non_nan_count}, unique: {unique_count})")
    
    # Analyze feature categories
    print(f"\n🏷️  Feature Categories:")
    
    # Basic features
    basic_features = [col for col in features.columns if col in 
                     ['current_h3_cell', 'current_speed', 'current_heading', 'lat', 'lon', 'time_in_current_cell']]
    print(f"   Basic State: {len(basic_features)} features")
    
    # Historical features  
    historical_features = [col for col in features.columns if any(x in col for x in ['_6h', '_12h', '_24h', 'visited', 'avg_'])]
    print(f"   Historical: {len(historical_features)} features")
    
    # Movement features
    movement_features = [col for col in features.columns if any(x in col for x in ['speed_', 'heading_', 'efficiency', 'trend'])]
    print(f"   Movement: {len(movement_features)} features")
    
    # Journey features
    journey_features = [col for col in features.columns if any(x in col for x in ['journey', 'distance', 'phase', 'cumulative'])]
    print(f"   Journey: {len(journey_features)} features")
    
    # Geographic features
    geo_features = [col for col in features.columns if any(x in col for x in ['coastal', 'depth', 'region', 'lane'])]
    print(f"   Geographic: {len(geo_features)} features")
    
    # Operational features
    operational_features = [col for col in features.columns if any(x in col for x in ['cargo', 'port', 'anchorage', 'hour', 'day', 'weekend'])]
    print(f"   Operational: {len(operational_features)} features")
    
    # Show sample of data
    print(f"\n📄 Sample data (first 3 rows, first 10 columns):")
    print(features.iloc[:3, :10])
    
    return features

if __name__ == "__main__":
    features = test_feature_extraction()
