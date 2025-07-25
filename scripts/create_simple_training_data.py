#!/usr/bin/env python3
"""
Create simple training sequences for H3 cell prediction
Very basic: input = current cell + features → target = next cell
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/home/marius/repo_linux/ais-forecasting/src')

from features.vessel_h3_tracker import VesselH3Tracker
from features.vessel_features import VesselFeatureExtractor

def create_simple_training_data():
    """Create the simplest possible training data for H3 cell prediction"""
    
    print("🚀 Creating Simple Training Data for H3 Cell Prediction...")
    
    # Load a small sample of data
    print("\n📊 Loading test data...")
    df = pd.read_pickle('/home/marius/repo_linux/ais-forecasting/data/raw/ais_cape_data_2024.pkl')
    
    # Take just one vessel with decent amount of data
    vessel_counts = df['imo'].value_counts()
    test_vessel = vessel_counts.index[0]  # Vessel with most records
    vessel_data = df[df['imo'] == test_vessel].head(200).copy()  # 200 records for sequences
    
    print(f"🚢 Using vessel {test_vessel}: {len(vessel_data)} records")
    
    # Step 1: Convert to H3 sequence and extract features
    print("\n🗺️  Converting to H3 and extracting features...")
    tracker = VesselH3Tracker(h3_resolution=5)
    h3_sequence = tracker.convert_vessel_to_h3_sequence(vessel_data)
    
    extractor = VesselFeatureExtractor(h3_resolution=5)
    features_df = extractor.extract_all_features(h3_sequence)
    
    print(f"   ✅ Features ready: {len(features_df)} positions")
    
    # Step 2: Create simple sequences
    print("\n🔗 Creating training sequences...")
    
    # Simple approach: each row predicts the next H3 cell
    sequences = []
    
    for i in range(len(features_df) - 1):  # -1 because we need next cell
        current_row = features_df.iloc[i]
        next_row = features_df.iloc[i + 1]
        
        # Input features (simplified for now)
        input_features = {
            'current_h3_cell': current_row['current_h3_cell'],
            'current_speed': current_row['current_speed'],
            'current_heading': current_row['current_heading'],
            'lat': current_row['lat'],
            'lon': current_row['lon'],
            'time_in_current_cell': current_row['time_in_current_cell']
        }
        
        # Target: next H3 cell
        target = next_row['current_h3_cell']
        
        # Combine
        sequence = {**input_features, 'target_h3_cell': target}
        sequences.append(sequence)
    
    # Convert to DataFrame
    training_df = pd.DataFrame(sequences)
    
    print(f"   ✅ Training sequences created: {len(training_df)} samples")
    print(f"   📊 Features: {list(training_df.columns[:-1])}")
    print(f"   🎯 Target: {training_df.columns[-1]}")
    
    # Step 3: Show sample
    print("\n📋 Sample training data:")
    print(training_df.head()[['current_h3_cell', 'current_speed', 'target_h3_cell']])
    
    # Step 4: Save the training data
    output_path = '/home/marius/repo_linux/ais-forecasting/data/processed/training_sets/simple_h3_sequences.pkl'
    training_df.to_pickle(output_path)
    
    print(f"\n💾 Training data saved to: {output_path}")
    print(f"🎯 Ready for machine learning! {len(training_df)} training examples")
    
    # Step 5: Quick analysis
    unique_cells = training_df['current_h3_cell'].nunique()
    unique_targets = training_df['target_h3_cell'].nunique()
    
    print(f"\n📊 Data Analysis:")
    print(f"   - Unique current cells: {unique_cells}")
    print(f"   - Unique target cells: {unique_targets}")
    print(f"   - Average speed: {training_df['current_speed'].mean():.1f} knots")
    
    return training_df

if __name__ == "__main__":
    try:
        result = create_simple_training_data()
        print("\n✅ Training data creation completed successfully!")
        print("🚀 Next step: Train a simple classifier to predict H3 cells!")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
