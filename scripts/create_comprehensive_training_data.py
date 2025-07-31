#!/usr/bin/env python3
"""
Phase 4: Create comprehensive training data using ALL 42 high-quality features
from VesselFeatureExtractor for H3 cell prediction.

This replaces the limited 6-feature approach with full feature utilization.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.vessel_h3_tracker import VesselH3Tracker
from features.vessel_features import VesselFeatureExtractor

def create_comprehensive_training_data(n_vessels=10, max_records_per_vessel=500):
    """
    Create comprehensive training data using ALL 42 high-quality features.
    
    Args:
        n_vessels: Number of vessels to include
        max_records_per_vessel: Maximum records per vessel to keep processing manageable
    """
    
    print("🚀 Phase 4: Creating Comprehensive Training Data with ALL Features...")
    print(f"   🚢 Target vessels: {n_vessels}")
    print(f"   📊 Max records per vessel: {max_records_per_vessel}")
    
    # Load data and select vessels
    print("\n📊 Loading AIS data...")
    df = pd.read_pickle('/home/marius/repo_linux/ais-forecasting/data/raw/ais_cape_data_2024.pkl')
    
    # Get vessels with substantial data
    vessel_counts = df['imo'].value_counts()
    selected_vessels = vessel_counts.head(n_vessels).index.tolist()
    
    print(f"   ✅ Selected {len(selected_vessels)} vessels with most data")
    print(f"   📈 Records per vessel: {vessel_counts.head(n_vessels).values}")
    
    # Process each vessel and combine
    all_sequences = []
    successful_vessels = 0
    
    for i, vessel_imo in enumerate(selected_vessels):
        try:
            print(f"\n🚢 Processing vessel {i+1}/{len(selected_vessels)}: {vessel_imo}")
            
            # Get vessel data
            vessel_data = df[df['imo'] == vessel_imo].head(max_records_per_vessel).copy()
            
            if len(vessel_data) < 10:  # Need minimum data for features
                print(f"   ⚠️  Skipping: only {len(vessel_data)} records")
                continue
            
            # Convert to H3 sequence
            tracker = VesselH3Tracker(h3_resolution=5)
            h3_sequence = tracker.convert_vessel_to_h3_sequence(vessel_data)
            
            if len(h3_sequence) < 5:  # Need minimum for sequences
                print(f"   ⚠️  Skipping: only {len(h3_sequence)} H3 positions")
                continue
            
            # Extract ALL features using VesselFeatureExtractor
            extractor = VesselFeatureExtractor(h3_resolution=5)
            features_df = extractor.extract_all_features(h3_sequence)
            
            if len(features_df) < 3:  # Need minimum for training sequences
                print(f"   ⚠️  Skipping: only {len(features_df)} feature rows")
                continue
            
            print(f"   ✅ Features extracted: {len(features_df)} positions, {len(features_df.columns)} features")
            
            # Create training sequences: each row predicts next H3 cell
            vessel_sequences = []
            
            for j in range(len(features_df) - 1):  # -1 because we need next cell
                current_row = features_df.iloc[j]
                next_row = features_df.iloc[j + 1]
                
                # Use ALL features as input (except target)
                input_features = current_row.to_dict()
                
                # Add vessel identifier for multi-vessel training
                input_features['vessel_imo'] = vessel_imo
                
                # Target: next H3 cell
                input_features['target_h3_cell'] = next_row['current_h3_cell']
                
                vessel_sequences.append(input_features)
            
            all_sequences.extend(vessel_sequences)
            successful_vessels += 1
            
            print(f"   ✅ Created {len(vessel_sequences)} training sequences")
            
        except Exception as e:
            print(f"   ❌ Failed processing vessel {vessel_imo}: {e}")
            continue
    
    if not all_sequences:
        raise ValueError("No training sequences created! Check data and processing.")
    
    # Convert to DataFrame
    print(f"\n🔗 Combining sequences from {successful_vessels} vessels...")
    training_df = pd.DataFrame(all_sequences)
    
    print(f"   ✅ Total sequences: {len(training_df)}")
    print(f"   ✅ Total features: {len(training_df.columns) - 1}")  # -1 for target
    print(f"   ✅ Unique vessels: {training_df['vessel_imo'].nunique()}")
    
    # Identify high-quality features (non-constant, good variance)
    print(f"\n🔍 Analyzing feature quality...")
    
    feature_quality = {}
    for col in training_df.columns:
        if col in ['target_h3_cell', 'vessel_imo']:
            continue
            
        non_null_count = training_df[col].count()
        unique_count = training_df[col].nunique()
        
        if unique_count <= 1:
            feature_quality[col] = "CONSTANT"
        elif unique_count <= 3:
            feature_quality[col] = "LIMITED"
        else:
            feature_quality[col] = "GOOD"
    
    good_features = [k for k, v in feature_quality.items() if v == "GOOD"]
    limited_features = [k for k, v in feature_quality.items() if v == "LIMITED"]
    constant_features = [k for k, v in feature_quality.items() if v == "CONSTANT"]
    
    print(f"   ✅ High-quality features: {len(good_features)}")
    print(f"   ⚠️  Limited-variance features: {len(limited_features)}")
    print(f"   ❌ Constant features: {len(constant_features)}")
    
    # Show feature categories
    print(f"\n📊 Feature breakdown:")
    print(f"   🎯 Good features ({len(good_features)}): {good_features[:10]}{'...' if len(good_features) > 10 else ''}")
    if limited_features:
        print(f"   ⚠️  Limited features ({len(limited_features)}): {limited_features}")
    if constant_features:
        print(f"   ❌ Constant features ({len(constant_features)}): {constant_features}")
    
    # Data quality summary
    print(f"\n📋 Training Data Summary:")
    print(f"   - Training sequences: {len(training_df):,}")
    print(f"   - Features available: {len(training_df.columns) - 2}")  # -2 for target and vessel_imo
    print(f"   - High-quality features: {len(good_features)}")
    print(f"   - Unique vessels: {training_df['vessel_imo'].nunique()}")
    print(f"   - Unique target cells: {training_df['target_h3_cell'].nunique()}")
    
    # Save comprehensive training data
    output_path = '/home/marius/repo_linux/ais-forecasting/data/processed/training_sets/comprehensive_h3_sequences.pkl'
    training_df.to_pickle(output_path)
    
    print(f"\n💾 Comprehensive training data saved to: {output_path}")
    
    # Save feature quality analysis
    feature_quality_df = pd.DataFrame([
        {'feature': k, 'quality': v, 'unique_values': training_df[k].nunique() if k in training_df.columns else 0}
        for k, v in feature_quality.items()
    ])
    
    quality_path = '/home/marius/repo_linux/ais-forecasting/data/processed/training_sets/feature_quality_analysis.pkl'
    feature_quality_df.to_pickle(quality_path)
    
    print(f"💾 Feature quality analysis saved to: {quality_path}")
    
    # Quick sample
    print(f"\n📋 Sample data (first 3 rows, key features):")
    sample_cols = ['vessel_imo', 'current_h3_cell', 'current_speed', 'lat', 'lon', 'target_h3_cell']
    available_cols = [col for col in sample_cols if col in training_df.columns]
    print(training_df[available_cols].head(3))
    
    return training_df, feature_quality

if __name__ == "__main__":
    try:
        # Create comprehensive training data
        training_data, feature_analysis = create_comprehensive_training_data(
            n_vessels=10,  # Start with 10 vessels for testing
            max_records_per_vessel=300  # Manageable size
        )
        
        print("\n🎉 SUCCESS! Comprehensive training data created!")
        print("🚀 Next step: Train models with ALL available features!")
        print("📈 Expected: Significant accuracy improvement over 6-feature baseline")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
