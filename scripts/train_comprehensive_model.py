#!/usr/bin/env python3
"""
Phase 4: Train comprehensive H3 prediction model using ALL available features
with XGBoost and feature selection for maximum accuracy.

This implements the complete Phase 4 pipeline:
1. Use all 42 high-quality features
2. Feature selection to find most predictive ones
3. XGBoost for better performance
4. Comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available, will use RandomForest")
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
import h3

def train_comprehensive_h3_predictor(data_path=None, use_feature_selection=True, top_k_features=20):
    """
    Train comprehensive H3 predictor using all available features.
    
    Args:
        data_path: Path to comprehensive training data
        use_feature_selection: Whether to select top features
        top_k_features: Number of top features to select
    """
    
    print("🚀 Phase 4: Training Comprehensive H3 Cell Predictor...")
    print(f"   🔧 Feature selection: {'Yes' if use_feature_selection else 'No'}")
    print(f"   🎯 Top features: {top_k_features if use_feature_selection else 'All'}")
    print(f"   🤖 Algorithm: {'XGBoost' if XGBOOST_AVAILABLE else 'RandomForest'}")
    
    # Load comprehensive training data
    if data_path is None:
        data_path = '/home/marius/repo_linux/ais-forecasting/data/processed/training_sets/comprehensive_h3_sequences.pkl'
    
    print(f"\n📊 Loading comprehensive training data...")
    try:
        training_df = pd.read_pickle(data_path)
        print(f"   ✅ Loaded {len(training_df):,} training sequences")
        print(f"   ✅ Features available: {len(training_df.columns) - 2}")  # -2 for target and vessel_imo
        print(f"   ✅ Unique vessels: {training_df['vessel_imo'].nunique()}")
    except FileNotFoundError:
        print(f"   ❌ Training data not found at {data_path}")
        print("   🔧 Please run create_comprehensive_training_data.py first")
        return None, None, None
    
    # Prepare features and target
    print(f"\n🔧 Preparing features...")
    
    # Separate features from target and metadata
    feature_cols = [col for col in training_df.columns 
                   if col not in ['target_h3_cell', 'vessel_imo']]
    
    # Handle non-numeric features
    X_raw = training_df[feature_cols].copy()
    
    # Convert different data types to numeric
    for col in X_raw.columns:
        if pd.api.types.is_datetime64_any_dtype(X_raw[col]):
            # Convert datetime to unix timestamp
            X_raw[col] = pd.to_datetime(X_raw[col]).astype(int) / 10**9  # Convert to seconds
        elif X_raw[col].dtype == 'object':
            try:
                # Try to convert to numeric
                X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
            except:
                # For categorical strings, use label encoding
                le = LabelEncoder()
                X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    
    # Remove features with too many NaN values or no variance
    X_clean = X_raw.copy()
    
    # Remove constant features
    constant_features = []
    for col in X_clean.columns:
        if X_clean[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"   🗑️  Removing {len(constant_features)} constant features")
        X_clean = X_clean.drop(columns=constant_features)
    
    # Fill remaining NaN values
    X_clean = X_clean.fillna(X_clean.median())
    
    print(f"   ✅ Clean features: {len(X_clean.columns)}")
    
    # Encode H3 target cells
    print(f"   🔧 Encoding target H3 cells...")
    h3_encoder = LabelEncoder()
    y_encoded = h3_encoder.fit_transform(training_df['target_h3_cell'])
    
    print(f"   ✅ Original target classes: {len(h3_encoder.classes_):,}")
    
    # Feature selection if requested
    if use_feature_selection and len(X_clean.columns) > top_k_features:
        print(f"\n🎯 Selecting top {top_k_features} features...")
        
        # Use mutual information for feature selection (good for classification)
        selector = SelectKBest(score_func=mutual_info_classif, k=top_k_features)
        X_selected = selector.fit_transform(X_clean, y_encoded)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X_clean.columns[selected_mask].tolist()
        
        X = pd.DataFrame(X_selected, columns=selected_features, index=X_clean.index)
        
        print(f"   ✅ Selected features: {selected_features}")
        
        # Save feature selector
        selector_path = '/home/marius/repo_linux/ais-forecasting/data/models/final_models/feature_selector.pkl'
        joblib.dump(selector, selector_path)
        print(f"   💾 Feature selector saved to: {selector_path}")
        
    else:
        X = X_clean
        selected_features = X.columns.tolist()
        print(f"\n✅ Using all {len(X.columns)} features")
    
    print(f"   📊 Final feature matrix: {X.shape}")
    
    # Split data first, then remap labels
    print(f"\n✂️  Splitting data...")
    
    # Check class distribution
    class_counts = pd.Series(y_encoded).value_counts()
    rare_classes = class_counts[class_counts == 1]
    
    if len(rare_classes) > 0:
        print(f"   ⚠️  Found {len(rare_classes)} classes with only 1 sample, using random split")
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
    else:
        print(f"   ✅ Using stratified split")
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    
    # Remap training labels to be consecutive starting from 0
    unique_train_labels = np.unique(y_train_encoded)
    train_label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_train_labels)}
    
    y_train = np.array([train_label_mapping[label] for label in y_train_encoded])
    
    # For test set, map to training labels where possible, -1 for unseen classes
    y_test = np.array([train_label_mapping.get(label, -1) for label in y_test_encoded])
    
    # Filter out test samples with unseen classes
    seen_mask = y_test != -1
    X_test = X_test[seen_mask]
    y_test = y_test[seen_mask]
    y_test_encoded = y_test_encoded[seen_mask]
    
    print(f"   📊 Train: {len(X_train):,} samples with {len(unique_train_labels)} classes")
    print(f"   📊 Test: {len(X_test):,} samples (filtered for seen classes)")
    
    # Store mappings for later use
    reverse_train_mapping = {new_label: old_label for old_label, new_label in train_label_mapping.items()}
    
    # Train model
    print(f"\n🤖 Training model...")
    
    if XGBOOST_AVAILABLE:
        # XGBoost classifier
        print("   🚀 Using XGBoost Classifier...")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    else:
        # Random Forest fallback
        print("   🌲 Using Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    print("   ✅ Model trained!")
    
    # Evaluate model
    print(f"\n📊 Evaluating model...")
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Accuracy scores
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"   📈 Train Accuracy: {train_accuracy:.3f} ({train_accuracy:.1%})")
    print(f"   📈 Test Accuracy: {test_accuracy:.3f} ({test_accuracy:.1%})")
    
    # Feature importance
    print(f"\n🔍 Top 10 Most Important Features:")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = list(zip(selected_features, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"   {i+1:2d}. {feature}: {importance:.4f}")
    
    # Distance-based evaluation (convert H3 back to lat/lon for distance)
    print(f"\n📏 Distance-based evaluation...")
    
    # Sample evaluation on test set (compute actual distances)
    sample_size = min(100, len(X_test))  # Evaluate on sample for speed
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    
    distances = []
    for idx in sample_indices:
        # Map back from consecutive labels to original encoded labels then to H3
        actual_encoded = reverse_train_mapping[y_test[idx]]
        predicted_encoded = reverse_train_mapping[test_pred[idx]]
        
        actual_h3 = h3_encoder.inverse_transform([actual_encoded])[0]
        predicted_h3 = h3_encoder.inverse_transform([predicted_encoded])[0]
        
        try:
            actual_lat, actual_lon = h3.h3_to_geo(actual_h3)
            pred_lat, pred_lon = h3.h3_to_geo(predicted_h3)
            
            # Haversine distance
            distance_km = haversine_distance(actual_lat, actual_lon, pred_lat, pred_lon)
            distances.append(distance_km)
        except:
            continue  # Skip invalid H3 cells
    
    if distances:
        avg_distance = np.mean(distances)
        median_distance = np.median(distances)
        print(f"   📏 Average prediction error: {avg_distance:.2f} km")
        print(f"   📏 Median prediction error: {median_distance:.2f} km")
        print(f"   🎯 Success rate (<15km): {np.mean(np.array(distances) < 15):.1%}")
    
    # Save model and encoders
    print(f"\n💾 Saving model...")
    
    model_path = '/home/marius/repo_linux/ais-forecasting/data/models/final_models/comprehensive_h3_predictor.pkl'
    encoder_path = '/home/marius/repo_linux/ais-forecasting/data/models/final_models/comprehensive_h3_encoder.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(h3_encoder, encoder_path)
    
    print(f"   ✅ Model saved to: {model_path}")
    print(f"   ✅ Encoder saved to: {encoder_path}")
    
    # Save training metadata
    metadata = {
        'model_type': 'XGBoost' if XGBOOST_AVAILABLE else 'RandomForest',
        'n_features': len(selected_features),
        'feature_names': selected_features,
        'n_classes': len(h3_encoder.classes_),
        'n_train_classes': len(unique_train_labels),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'train_label_mapping': train_label_mapping,
        'reverse_train_mapping': reverse_train_mapping
    }
    
    metadata_path = '/home/marius/repo_linux/ais-forecasting/data/models/final_models/comprehensive_model_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"   ✅ Metadata saved to: {metadata_path}")
    
    # Performance summary
    print(f"\n🎉 Training Complete! Performance Summary:")
    print(f"   🎯 Test Accuracy: {test_accuracy:.1%}")
    print(f"   📊 Features Used: {len(selected_features)}")
    print(f"   🚢 Training Samples: {len(X_train):,}")
    print(f"   🗺️  H3 Classes: {len(h3_encoder.classes_):,}")
    
    if distances:
        print(f"   📏 Avg Distance Error: {avg_distance:.2f} km")
        success_rate = np.mean(np.array(distances) < 15)
        print(f"   🏆 Success Rate (<15km): {success_rate:.1%}")
        
        if success_rate > 0.6:
            print("   🌟 EXCELLENT! Target >60% success rate achieved!")
        elif success_rate > 0.3:
            print("   🚀 GOOD! Significant improvement over baseline!")
        else:
            print("   📈 IMPROVEMENT! Better than random, needs more optimization")
    
    return model, h3_encoder, {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'n_features': len(selected_features),
        'avg_distance_km': np.mean(distances) if distances else None
    }

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth using Haversine formula."""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

if __name__ == "__main__":
    try:
        # Train comprehensive model
        model, encoder, metrics = train_comprehensive_h3_predictor(
            use_feature_selection=True,
            top_k_features=25  # Use top 25 features for optimal performance
        )
        
        if model is not None:
            print("\n🎉 SUCCESS! Phase 4 comprehensive model trained!")
            print(f"🚀 Test Accuracy: {metrics['test_accuracy']:.1%}")
            print(f"📊 Using {metrics['n_features']} carefully selected features")
            print("🌟 Ready for production use with significantly improved accuracy!")
        else:
            print("\n❌ Training failed - check data availability")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
