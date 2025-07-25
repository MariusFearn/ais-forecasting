"""
Create Training Data for Vessel H3 Cell Prediction

This script converts Phase 2 vessel features into ML training datasets.

Goal: Transform vessel journey data into:
- Input: Current vessel state (65 features)  
- Target: Next H3 cell the vessel actually visited

Usage:
    python scripts/create_training_data.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.vessel_h3_tracker import VesselH3Tracker
from features.vessel_features import VesselFeatureExtractor

def load_sample_vessel_data():
    """Load sample vessel data for testing"""
    data_path = Path("data/raw/ais_cape_data_2024.pkl")
    
    print(f"Loading data from {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    
    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['mdt'].min()} to {df['mdt'].max()}")
    print(f"Unique vessels: {df['imo'].nunique()}")
    
    return df

def create_vessel_sequences(df, max_vessels=5):
    """Create H3 sequences for multiple vessels"""
    
    tracker = VesselH3Tracker(h3_resolution=5)
    extractor = VesselFeatureExtractor()
    
    all_sequences = []
    vessels_processed = 0
    
    print(f"\nProcessing up to {max_vessels} vessels...")
    
    for imo in df['imo'].unique():
        if vessels_processed >= max_vessels:
            break
            
        vessel_data = df[df['imo'] == imo].copy()
        vessel_data = vessel_data.sort_values('mdt')
        
        print(f"Processing vessel {imo}: {len(vessel_data)} records")
        
        try:
            # Convert to H3 sequence
            h3_sequence = tracker.convert_vessel_to_h3_sequence(vessel_data)
            
            if len(h3_sequence) < 10:  # Skip short sequences
                print(f"  Skipping vessel {imo}: sequence too short ({len(h3_sequence)})")
                continue
            
            # Extract features for each point in sequence
            features_df = extractor.extract_features(vessel_data, h3_sequence)
            
            if features_df is not None and len(features_df) > 0:
                features_df['imo'] = imo
                all_sequences.append(features_df)
                vessels_processed += 1
                print(f"  ✅ Extracted {len(features_df)} feature rows")
            else:
                print(f"  ❌ Feature extraction failed")
                
        except Exception as e:
            print(f"  ❌ Error processing vessel {imo}: {str(e)}")
            continue
    
    if not all_sequences:
        raise ValueError("No valid vessel sequences created")
    
    # Combine all sequences
    combined_df = pd.concat(all_sequences, ignore_index=True)
    print(f"\n✅ Total feature rows: {len(combined_df)}")
    print(f"✅ Vessels processed: {vessels_processed}")
    
    return combined_df

def create_training_pairs(features_df):
    """Create input-target pairs for ML training"""
    
    print("\nCreating training pairs...")
    
    training_pairs = []
    
    for imo in features_df['imo'].unique():
        vessel_features = features_df[features_df['imo'] == imo].copy()
        vessel_features = vessel_features.sort_values('timestamp')
        
        # Create input-target pairs
        for i in range(len(vessel_features) - 1):
            current_row = vessel_features.iloc[i]
            next_row = vessel_features.iloc[i + 1]
            
            # Input: current features (exclude target columns)
            feature_columns = [col for col in vessel_features.columns 
                             if col not in ['imo', 'timestamp', 'current_h3_cell']]
            
            input_features = current_row[feature_columns].to_dict()
            target_cell = next_row['current_h3_cell']
            
            training_pairs.append({
                'imo': imo,
                'timestamp': current_row['timestamp'],
                'features': input_features,
                'target_h3_cell': target_cell,
                'current_h3_cell': current_row['current_h3_cell']
            })
    
    print(f"✅ Created {len(training_pairs)} training pairs")
    
    return training_pairs

def save_training_data(training_pairs):
    """Save training data to processed directory"""
    
    output_dir = Path("data/processed/training_sets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "vessel_next_cell_training.pkl"
    
    print(f"\nSaving training data to {output_file}")
    
    with open(output_file, 'wb') as f:
        pickle.dump(training_pairs, f)
    
    print(f"✅ Saved {len(training_pairs)} training pairs")
    
    # Create summary
    df_summary = pd.DataFrame(training_pairs)
    print(f"\nTraining Data Summary:")
    print(f"- Total pairs: {len(df_summary)}")
    print(f"- Unique vessels: {df_summary['imo'].nunique()}")
    print(f"- Unique target cells: {df_summary['target_h3_cell'].nunique()}")
    print(f"- Date range: {df_summary['timestamp'].min()} to {df_summary['timestamp'].max()}")
    
    return output_file

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("CREATING TRAINING DATA FOR VESSEL H3 PREDICTION")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        df = load_sample_vessel_data()
        
        # Step 2: Create vessel sequences with features
        features_df = create_vessel_sequences(df, max_vessels=3)  # Start small
        
        # Step 3: Create training pairs
        training_pairs = create_training_pairs(features_df)
        
        # Step 4: Save training data
        output_file = save_training_data(training_pairs)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Training data created.")
        print(f"Next step: Train model using {output_file}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
