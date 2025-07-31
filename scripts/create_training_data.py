#!/usr/bin/env python3
"""
Unified Data Creation Script

Configurable data creation script that can handle all data generation scenarios:
- Simple (Phase 1): Single vessel, basic features
- Comprehensive (Phase 4): Multi-vessel, all features  
- Massive (Phase 5): All data, maximum scale

Usage:
    python scripts/create_training_data.py --config simple_data_creation
    python scripts/create_training_data.py --config comprehensive_data_creation  
    python scripts/create_training_data.py --config massive_data_creation
"""

import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.vessel_h3_tracker import VesselH3Tracker
from features.vessel_features import VesselFeatureExtractor

def load_config(config_name):
    """Load data creation configuration from YAML file."""
    config_path = Path(f"config/experiment_configs/{config_name}.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load default config if specified
    if 'defaults' in config:
        default_path = Path("config/default.yaml")
        if default_path.exists():
            with open(default_path, 'r') as f:
                default_config = yaml.safe_load(f)
            
            # Merge configs
            def merge_configs(default, override):
                result = default.copy()
                for key, value in override.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = merge_configs(result[key], value)
                    else:
                        result[key] = value
                return result
            
            config = merge_configs(default_config, config)
    
    return config

def load_and_combine_data(config):
    """Load and combine data files according to configuration."""
    print("📊 Loading AIS data...")
    
    data_files = config['data_source']['data_files']
    all_data = []
    
    # Use progress bar for multiple files
    if len(data_files) > 1:
        file_iterator = tqdm(data_files, desc="Loading data files")
    else:
        file_iterator = data_files
    
    for file_path in file_iterator:
        try:
            if not Path(file_path).exists():
                print(f"   ⚠️  Skipping missing file: {file_path}")
                continue
                
            df = pd.read_pickle(file_path)
            
            # Add data year if configured
            if config.get('processing', {}).get('include_data_year', False):
                year = Path(file_path).stem.split('_')[-1]
                df['data_year'] = int(year)
            
            all_data.append(df)
            print(f"   ✅ {Path(file_path).name}: {len(df):,} records")
            
        except Exception as e:
            print(f"   ❌ Failed to load {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data files loaded successfully!")
    
    # Combine all data
    if len(all_data) > 1:
        print("🔗 Combining data files...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp if configured
        if config.get('processing', {}).get('sort_by_timestamp', False):
            print("   📅 Sorting by timestamp...")
            combined_df = combined_df.sort_values('mdt').reset_index(drop=True)
    else:
        combined_df = all_data[0]
    
    print(f"   ✅ Total combined records: {len(combined_df):,}")
    return combined_df

def select_vessels(df, config):
    """Select vessels according to configuration."""
    print("🚢 Selecting vessels...")
    
    vessel_config = config['data_source']['vessel_selection']
    method = vessel_config['method']
    max_vessels = vessel_config.get('max_vessels')
    min_records = vessel_config['min_records_per_vessel']
    
    # Analyze vessel distribution
    vessel_counts = df['imo'].value_counts()
    
    if method == "top_records":
        # Select vessels with most records
        qualified_vessels = vessel_counts[vessel_counts >= min_records]
        
        if max_vessels:
            selected_vessels = qualified_vessels.head(max_vessels).index.tolist()
        else:
            selected_vessels = qualified_vessels.index.tolist()
            
    elif method == "qualified_vessels":
        # All vessels meeting minimum criteria
        qualified_vessels = vessel_counts[vessel_counts >= min_records]
        
        if max_vessels and len(qualified_vessels) > max_vessels:
            selected_vessels = qualified_vessels.head(max_vessels).index.tolist()
        else:
            selected_vessels = qualified_vessels.index.tolist()
    else:
        raise ValueError(f"Unknown vessel selection method: {method}")
    
    print(f"   ✅ Selected {len(selected_vessels)} vessels")
    print(f"   📈 Records per vessel: {vessel_counts[selected_vessels].head(5).values}")
    
    return selected_vessels

def extract_features_for_vessel(vessel_data, config):
    """Extract features for a single vessel."""
    processing_config = config['processing']
    
    # Convert to H3 sequence
    tracker = VesselH3Tracker(h3_resolution=processing_config['h3_resolution'])
    h3_sequence = tracker.convert_vessel_to_h3_sequence(vessel_data)
    
    if len(h3_sequence) < processing_config['min_h3_positions']:
        return None, f"Only {len(h3_sequence)} H3 positions"
    
    # Extract features
    extractor = VesselFeatureExtractor(h3_resolution=processing_config['h3_resolution'])
    
    feature_extraction = processing_config['feature_extraction']
    if feature_extraction == "basic":
        # For simple data creation, extract only basic features
        features_df = extractor.extract_all_features(h3_sequence)
        # Keep only basic columns for simple mode
        if config['features']['feature_set'] == "simple":
            basic_cols = ['current_h3_cell', 'current_speed', 'current_heading', 
                         'lat', 'lon', 'time_in_current_cell']
            available_cols = [col for col in basic_cols if col in features_df.columns]
            features_df = features_df[available_cols].copy()
    else:
        # Comprehensive feature extraction
        features_df = extractor.extract_all_features(h3_sequence)
    
    if len(features_df) < processing_config['min_feature_rows']:
        return None, f"Only {len(features_df)} feature rows"
    
    return features_df, None

def create_training_sequences(features_df, vessel_imo, config):
    """Create training sequences from features."""
    sequences = []
    
    for i in range(len(features_df) - 1):  # -1 because we need next cell
        current_row = features_df.iloc[i]
        next_row = features_df.iloc[i + 1]
        
        # Use features as input
        if config['features']['feature_set'] == "simple":
            # Simple mode: specific features only
            input_features = {
                'current_h3_cell': current_row['current_h3_cell'],
                'current_speed': current_row['current_speed'],
                'current_heading': current_row['current_heading'],
                'lat': current_row['lat'],
                'lon': current_row['lon'],
                'time_in_current_cell': current_row['time_in_current_cell']
            }
        else:
            # Comprehensive mode: all features
            input_features = current_row.to_dict()
        
        # Add vessel ID if configured
        if config['features']['include_vessel_id']:
            input_features['vessel_imo'] = vessel_imo
        
        # Add data year if available
        if 'data_year' in current_row:
            input_features['data_year'] = current_row['data_year']
        
        # Target: next H3 cell
        input_features['target_h3_cell'] = next_row['current_h3_cell']
        
        sequences.append(input_features)
    
    return sequences

def analyze_feature_quality(training_df, config):
    """Analyze feature quality if configured."""
    if not config.get('quality', {}).get('analyze_feature_quality', False):
        return None
    
    print("🔍 Analyzing feature quality...")
    
    thresholds = config['quality']['feature_quality_thresholds']
    constant_threshold = thresholds['constant_features']
    limited_threshold = thresholds['limited_features']
    
    feature_quality = {}
    exclude_cols = ['target_h3_cell', 'vessel_imo', 'data_year']
    
    for col in training_df.columns:
        if col in exclude_cols:
            continue
            
        unique_count = training_df[col].nunique()
        
        if unique_count <= constant_threshold:
            feature_quality[col] = "CONSTANT"
        elif unique_count <= limited_threshold:
            feature_quality[col] = "LIMITED"
        else:
            feature_quality[col] = "GOOD"
    
    good_features = [k for k, v in feature_quality.items() if v == "GOOD"]
    limited_features = [k for k, v in feature_quality.items() if v == "LIMITED"]
    constant_features = [k for k, v in feature_quality.items() if v == "CONSTANT"]
    
    print(f"   ✅ High-quality features: {len(good_features)}")
    print(f"   ⚠️  Limited-variance features: {len(limited_features)}")
    print(f"   ❌ Constant features: {len(constant_features)}")
    
    if good_features:
        print(f"   🎯 Good features: {good_features[:10]}{'...' if len(good_features) > 10 else ''}")
    
    return feature_quality

def save_outputs(training_df, feature_quality, config):
    """Save training data and analysis according to configuration."""
    print("💾 Saving outputs...")
    
    output_config = config['output']
    
    # Ensure output directory exists
    output_path = Path(output_config['data_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save training data
    training_df.to_pickle(output_config['data_path'])
    print(f"   ✅ Training data saved to: {output_config['data_path']}")
    
    # Save feature analysis if configured
    if output_config.get('save_feature_analysis', False) and feature_quality:
        feature_quality_df = pd.DataFrame([
            {'feature': k, 'quality': v, 'unique_values': training_df[k].nunique() if k in training_df.columns else 0}
            for k, v in feature_quality.items()
        ])
        
        analysis_path = output_config['feature_analysis_path']
        feature_quality_df.to_pickle(analysis_path)
        print(f"   ✅ Feature analysis saved to: {analysis_path}")
    
    # Show sample preview if configured
    if output_config.get('include_sample_preview', False):
        print("\n📋 Sample data (first 3 rows, key features):")
        sample_cols = ['vessel_imo', 'current_h3_cell', 'current_speed', 'lat', 'lon', 'target_h3_cell']
        if 'data_year' in training_df.columns:
            sample_cols.insert(1, 'data_year')
        available_cols = [col for col in sample_cols if col in training_df.columns]
        print(training_df[available_cols].head(3))
    
    # Temporal analysis if configured
    if output_config.get('include_temporal_analysis', False) and 'data_year' in training_df.columns:
        print("\n📅 Temporal distribution:")
        yearly_counts = training_df['data_year'].value_counts().sort_index()
        for year, count in yearly_counts.items():
            print(f"   {year}: {count:,} sequences ({count/len(training_df):.1%})")

def create_training_data(config_name):
    """Main data creation function."""
    print(f"🚀 Starting Data Creation with config: {config_name}")
    
    # Load configuration
    config = load_config(config_name)
    print(f"   📋 Experiment: {config['experiment']['name']}")
    print(f"   📋 Description: {config['experiment']['description']}")
    print(f"   📋 Phase: {config['experiment']['phase']}")
    
    # Load and combine data
    combined_df = load_and_combine_data(config)
    
    # Select vessels
    selected_vessels = select_vessels(combined_df, config)
    
    # Process each vessel
    print(f"\n🔄 Processing {len(selected_vessels)} vessels...")
    
    all_sequences = []
    successful_vessels = 0
    processing_config = config['processing']
    
    # Use progress bar for multiple vessels
    if len(selected_vessels) > 1:
        vessel_iterator = tqdm(selected_vessels, desc="Processing vessels")
    else:
        vessel_iterator = selected_vessels
    
    for vessel_imo in vessel_iterator:
        try:
            if len(selected_vessels) == 1:
                print(f"🚢 Processing vessel: {vessel_imo}")
            
            # Get vessel data
            vessel_data = combined_df[combined_df['imo'] == vessel_imo].copy()
            
            # Limit records per vessel
            max_records = processing_config['max_records_per_vessel']
            if len(vessel_data) > max_records:
                vessel_data = vessel_data.head(max_records)
            
            if len(vessel_data) < config['data_source']['vessel_selection']['min_records_per_vessel']:
                continue
            
            # Extract features
            features_df, error = extract_features_for_vessel(vessel_data, config)
            
            if features_df is None:
                if len(selected_vessels) == 1:
                    print(f"   ⚠️  Skipping: {error}")
                continue
            
            # Create training sequences
            vessel_sequences = create_training_sequences(features_df, vessel_imo, config)
            
            if len(vessel_sequences) < config['quality']['min_sequences_per_vessel']:
                continue
            
            all_sequences.extend(vessel_sequences)
            successful_vessels += 1
            
            if len(selected_vessels) == 1:
                print(f"   ✅ Features extracted: {len(features_df)} positions, {len(features_df.columns)} features")
                print(f"   ✅ Created {len(vessel_sequences)} training sequences")
            
        except Exception as e:
            if config['quality']['skip_vessels_with_errors']:
                continue
            else:
                raise e
    
    if not all_sequences:
        raise ValueError("No training sequences created! Check data and processing configuration.")
    
    # Convert to DataFrame
    print(f"\n🔗 Creating training dataset...")
    training_df = pd.DataFrame(all_sequences)
    
    print(f"   ✅ Total sequences: {len(training_df):,}")
    print(f"   ✅ Total features: {len(training_df.columns) - 1}")  # -1 for target
    print(f"   ✅ Successful vessels: {successful_vessels}")
    print(f"   ✅ Unique target cells: {training_df['target_h3_cell'].nunique()}")
    
    # Analyze feature quality
    feature_quality = analyze_feature_quality(training_df, config)
    
    # Save outputs
    save_outputs(training_df, feature_quality, config)
    
    print(f"\n🎉 Data creation completed successfully!")
    print(f"   📊 Created {len(training_df):,} training sequences")
    print(f"   🚢 From {successful_vessels} vessels")
    print(f"   🚀 Ready for training!")
    
    return training_df, feature_quality

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Create training data with configuration')
    parser.add_argument('--config', 
                        help='Configuration name (e.g., simple_data_creation)')
    parser.add_argument('--list-configs', action='store_true',
                        help='List available data creation configurations')
    
    args = parser.parse_args()
    
    if args.list_configs:
        config_dir = Path("config/experiment_configs")
        configs = [f.stem for f in config_dir.glob("*_data_creation.yaml")]
        print("📋 Available data creation configurations:")
        for config in sorted(configs):
            print(f"   • {config}")
        return
    
    if not args.config:
        parser.error("--config is required unless using --list-configs")
    
    try:
        training_df, feature_quality = create_training_data(args.config)
        print(f"\n✅ SUCCESS! Created {len(training_df):,} training sequences!")
        
    except Exception as e:
        print(f"\n❌ Data creation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
