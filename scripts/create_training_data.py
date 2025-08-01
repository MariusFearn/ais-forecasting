#!/usr/bin/env python3
"""
Unified Data Creation Script

Configurable data creation script that can handle all data generation scenarios:
- Simple (Phase 1): Single vessel, basic features
- Comprehensive (Phase 4): Multi-vessel, all features  
- Massive (Phase 5): All data, maximum scale

Usage:
    python scripts/create_training_data.py --config creation_data_simple
    python scripts/create_training_data.py --config creation_data_comprehensive  
    python scripts/create_training_data.py --config creation_data_massive
"""

import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import re
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.loader import AISDataLoader
from features.vessel_h3_tracker import VesselH3Tracker
from features.vessel_features import VesselFeatureExtractor

def load_config(config_name):
    """Load data creation configuration from YAML file with proper defaults inheritance."""
    config_path = Path(f"config/experiment_configs/{config_name}.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    def load_config_recursive(config_path, visited=None):
        """Recursively load configurations with defaults."""
        if visited is None:
            visited = set()
        
        # Prevent infinite loops
        if str(config_path) in visited:
            return {}
        visited.add(str(config_path))
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'defaults' not in config:
            return config
        
        # Start with empty result
        result = {}
        
        # Load each default file
        for default_file in config['defaults']:
            if default_file.startswith('../'):
                # Load from parent directory (config/)
                default_path = config_path.parent.parent / f"{default_file[3:]}.yaml"
            else:
                # Load from same directory (experiment_configs/)
                default_path = config_path.parent / f"{default_file}.yaml"
            
            if default_path.exists():
                default_config = load_config_recursive(default_path, visited.copy())
                result = merge_configs(result, default_config)
        
        # Merge current config (without defaults key)
        current_config = {k: v for k, v in config.items() if k != 'defaults'}
        result = merge_configs(result, current_config)
        
        return result
    
    def merge_configs(base, override):
        """Deep merge two configurations."""
        if base is None:
            return override
        if override is None:
            return base
        
        if not isinstance(base, dict) or not isinstance(override, dict):
            return override
        
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    return load_config_recursive(config_path)

def load_and_combine_data(config):
    """Load and combine data files according to configuration using AISDataLoader."""
    print("📊 Loading AIS data...")

    data_source_config = config['data_source']
    data_files = data_source_config.get('data_files', [])
    
    if not data_files:
        raise ValueError("Configuration missing 'data_files' under 'data_source'.")

    # Extract years from file paths, e.g., 'data/raw/ais_cape_data_2023.pkl' -> '2023'
    # This regex is safer and handles different path formats.
    years = sorted(list(set(re.findall(r'(\d{4})', " ".join(data_files)))))
    
    if not years:
        raise ValueError("Could not extract any years from the data file paths. Ensure they follow a format like 'ais_cape_data_YYYY.pkl'.")

    # Initialize the loader
    # The 'data' directory is the parent of 'raw' and 'processed'
    data_dir = str(Path(data_files[0]).resolve().parent.parent)
    use_duckdb = config.get('duckdb', {}).get('enabled', True)
    loader = AISDataLoader(data_dir=data_dir, use_duckdb=use_duckdb)

    print(f"   🔄 Using {'DuckDB' if use_duckdb else 'pandas'} backend for years: {', '.join(years)}")
    
    # Load data using the optimized loader
    # Pass down any filters if they exist in the config
    filters = data_source_config.get('filters')
    combined_df = loader.load_multi_year_data_optimized(years, filters=filters)

    # The new loader already handles combining and can apply filters.
    # Sorting by timestamp can be done here if needed.
    if config.get('processing', {}).get('sort_by_timestamp', False):
        print("   📅 Sorting by timestamp...")
        combined_df = combined_df.sort_values('mdt').reset_index(drop=True)

    # Add data year if configured. This is more robust than using filenames.
    if config.get('processing', {}).get('include_data_year', False):
        print("   📅 Adding data_year from 'mdt' timestamp...")
        # Ensure 'mdt' is datetime
        if not np.issubdtype(combined_df['mdt'].dtype, np.datetime64):
             combined_df['mdt'] = pd.to_datetime(combined_df['mdt'])
        combined_df['data_year'] = combined_df['mdt'].dt.year

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
                        help='Configuration name (e.g., creation_data_simple)')
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
