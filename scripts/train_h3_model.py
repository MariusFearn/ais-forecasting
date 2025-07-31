#!/usr/bin/env python3
"""
Unified H3 Prediction Training Script

Configurable training script that can handle all experiment types:
- Simple (Phase 1): RandomForest with 6 basic features
- Comprehensive (Phase 4): XGBoost with 42 features + selection  
- Massive (Phase 5): Large-scale training with all data

Usage:
    python scripts/train_h3_model.py --config simple_h3_experiment
    python scripts/train_h3_model.py --config comprehensive_h3_experiment  
    python scripts/train_h3_model.py --config massive_h3_experiment
"""

import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import yaml
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# ML imports
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Phase 5 fixes
try:
    from src.data.preprocessing import fix_datetime_categorical_issues
    PHASE5_FIXES_AVAILABLE = True
except ImportError:
    print("⚠️  Phase 5 fixes not available")
    PHASE5_FIXES_AVAILABLE = False

def load_config(config_name):
    """Load experiment configuration from YAML file."""
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
            
            # Merge configs (experiment config overrides defaults)
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

def prepare_features(df, config):
    """Prepare features according to configuration."""
    print("🔧 Preparing features...")
    
    # Apply Phase 5 fixes if enabled
    if config.get('training', {}).get('apply_phase5_fixes', False) and PHASE5_FIXES_AVAILABLE:
        print("   🔧 Applying Phase 5 data type fixes...")
        df = fix_datetime_categorical_issues(df)
    
    # Handle simple vs comprehensive feature preparation
    if 'features' in config['data']:
        # Simple mode: use specified features
        feature_cols = [col for col in config['data']['features'] if col != 'current_h3_encoded']
        X = df[feature_cols].copy()
        
        # Add encoded H3 if specified
        if 'current_h3_encoded' in config['data']['features']:
            h3_encoder = LabelEncoder()
            all_h3_cells = list(set(df['current_h3_cell'].tolist() + df[config['data']['target']].tolist()))
            h3_encoder.fit(all_h3_cells)
            X['current_h3_encoded'] = h3_encoder.transform(df['current_h3_cell'])
    else:
        # Comprehensive mode: use all features except excluded
        exclude_cols = config['data'].get('exclude_columns', ['target_h3_cell'])
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        
        # Convert categorical columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                except:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill missing values and handle inf
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], 0)
        
        # Remove constant features
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"   🧹 Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
    
    y = df[config['data']['target']].copy()
    
    print(f"   ✅ Features prepared: {len(X.columns)} features")
    return X, y

def select_features(X, y, config):
    """Apply feature selection if configured."""
    if not config.get('training', {}).get('use_feature_selection', False):
        return X, list(X.columns)
    
    top_k = config['training'].get('top_k_features', 25)
    if len(X.columns) <= top_k:
        print(f"   ℹ️  Skipping feature selection ({len(X.columns)} <= {top_k})")
        return X, list(X.columns)
    
    print(f"   🎯 Selecting top {top_k} features...")
    
    method = config.get('feature_selection', {}).get('method', 'mutual_info_classif')
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=top_k)
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"   ✅ Selected features: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}")
    return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

def create_model(config):
    """Create model based on configuration."""
    model_type = config['model']['type']
    params = config['model']['parameters']
    
    if model_type == 'xgboost':
        print(f"   🚀 Creating XGBoost Classifier...")
        return xgb.XGBClassifier(**params)
    else:
        print(f"   🌲 Creating RandomForest Classifier...")
        return RandomForestClassifier(**params)

def evaluate_model(model, X_test, y_test, h3_encoder, config):
    """Evaluate model with configured metrics."""
    print("\n📊 Evaluating model...")
    
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"   📈 Test Accuracy: {test_accuracy:.3f} ({test_accuracy:.1%})")
    
    results = {'test_accuracy': test_accuracy}
    
    # Feature importance
    if config['evaluation'].get('include_feature_importance', False) and hasattr(model, 'feature_importances_'):
        print("\n🔍 Top 10 Feature Importance:")
        feature_names = X_test.columns
        importances = model.feature_importances_
        for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {name}: {importance:.3f}")
        
        results['feature_importance'] = dict(zip(feature_names, importances))
    
    # Distance-based evaluation
    if config['evaluation'].get('include_distance_evaluation', False):
        print("\n📏 Distance-based evaluation...")
        try:
            sample_size = min(config['evaluation'].get('distance_sample_size', 500), len(X_test))
            sample_indices = np.random.choice(len(X_test), size=sample_size, replace=False)
            
            distances = []
            for i in sample_indices:
                actual_h3 = h3_encoder.inverse_transform([y_test[i]])[0]
                predicted_h3 = h3_encoder.inverse_transform([test_pred[i]])[0]
                
                # Simple distance calculation (placeholder)
                distance = 0 if actual_h3 == predicted_h3 else 1
                distances.append(distance)
            
            avg_distance = np.mean(distances)
            print(f"   📏 Average prediction distance: {avg_distance:.3f}")
            results['avg_distance'] = avg_distance
        except Exception as e:
            print(f"   ⚠️  Distance evaluation failed: {e}")
    
    # Sample prediction test
    if config['evaluation'].get('sample_prediction_test', False) and len(X_test) > 0:
        print("\n🔮 Sample prediction test...")
        sample_input = X_test.iloc[0:1]
        predicted_encoded = model.predict(sample_input)[0]
        predicted_h3 = h3_encoder.inverse_transform([predicted_encoded])[0]
        actual_h3 = h3_encoder.inverse_transform([y_test[0]])[0]
        
        print(f"   📍 Sample prediction:")
        print(f"      Predicted H3: {predicted_h3}")
        print(f"      Actual H3: {actual_h3}")
        print(f"      Correct: {'✅' if predicted_h3 == actual_h3 else '❌'}")
    
    return results

def save_model_and_metadata(model, h3_encoder, selected_features, results, config):
    """Save model, encoder, and metadata."""
    print("\n💾 Saving model and metadata...")
    
    # Create output directory
    model_dir = Path(config['output']['model_path']).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and encoder
    joblib.dump(model, config['output']['model_path'])
    joblib.dump(h3_encoder, config['output']['encoder_path'])
    
    # Save metadata
    metadata = {
        'experiment_name': config['experiment']['name'],
        'experiment_phase': config['experiment']['phase'],
        'description': config['experiment']['description'],
        'model_type': config['model']['type'],
        'features_used': selected_features,
        'n_features': len(selected_features),
        'training_timestamp': datetime.now().isoformat(),
        'results': results,
        'config': config
    }
    
    joblib.dump(metadata, config['output']['metadata_path'])
    
    print(f"   ✅ Model saved to: {config['output']['model_path']}")
    print(f"   ✅ Encoder saved to: {config['output']['encoder_path']}")  
    print(f"   ✅ Metadata saved to: {config['output']['metadata_path']}")

def train_h3_predictor(config_name):
    """Main training function."""
    print(f"🚀 Starting H3 Prediction Training with config: {config_name}")
    
    # Load configuration
    config = load_config(config_name)
    print(f"   📋 Experiment: {config['experiment']['name']}")
    print(f"   📋 Description: {config['experiment']['description']}")
    print(f"   📋 Phase: {config['experiment']['phase']}")
    
    # Load training data
    print(f"\n📊 Loading training data...")
    data_path = config['data']['training_data_path']
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    df = pd.read_pickle(data_path)
    print(f"   ✅ Loaded {len(df):,} training examples")
    
    # Prepare features and target
    X, y = prepare_features(df, config)
    
    # Encode target
    print(f"   🔧 Encoding target H3 cells...")
    h3_encoder = LabelEncoder()
    y_encoded = h3_encoder.fit_transform(y)
    print(f"   ✅ Target classes: {len(h3_encoder.classes_):,}")
    
    # Feature selection
    X_selected, selected_features = select_features(X, y_encoded, config)
    
    # Split data
    print(f"\n✂️  Splitting data...")
    test_size = config['data'].get('test_size', 0.2)
    random_state = config['data'].get('random_state', 42)
    
    # Handle single sample classes if configured
    if config.get('training', {}).get('handle_single_sample_classes', False):
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        single_sample_classes = np.sum(class_counts == 1)
        
        if single_sample_classes > len(unique_classes) * 0.1:
            print(f"   ⚠️  Many single-sample classes ({single_sample_classes}), using stratified split")
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_encoded, test_size=test_size, random_state=random_state, shuffle=True
            )
            # Filter test set to only include classes seen in training
            train_classes = set(y_train)
            test_mask = np.isin(y_test, list(train_classes))
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=test_size, random_state=random_state
        )
    
    print(f"   📊 Train: {len(X_train):,} samples")
    print(f"   📊 Test: {len(X_test):,} samples")
    
    # Create and train model
    print(f"\n🤖 Training model...")
    model = create_model(config)
    model.fit(X_train, y_train)
    print(f"   ✅ Model trained!")
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test, h3_encoder, config)
    
    # Save everything
    save_model_and_metadata(model, h3_encoder, selected_features, results, config)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"   🎯 Final Accuracy: {results['test_accuracy']:.1%}")
    print(f"   🔧 Features Used: {len(selected_features)}")
    print(f"   🚀 Model ready for predictions!")
    
    return model, h3_encoder, results

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Train H3 prediction model with configuration')
    parser.add_argument('--config', 
                        help='Configuration name (e.g., simple_h3_experiment)')
    parser.add_argument('--list-configs', action='store_true',
                        help='List available configurations')
    
    args = parser.parse_args()
    
    if args.list_configs:
        config_dir = Path("config/experiment_configs")
        configs = [f.stem for f in config_dir.glob("*_experiment.yaml")]
        print("📋 Available experiment configurations:")
        for config in sorted(configs):
            print(f"   • {config}")
        return
    
    if not args.config:
        parser.error("--config is required unless using --list-configs")
    
    try:
        model, encoder, results = train_h3_predictor(args.config)
        print(f"\n✅ SUCCESS! Model trained with {results['test_accuracy']:.1%} accuracy!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
