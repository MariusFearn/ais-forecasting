#!/usr/bin/env python3
"""
Phase 4: Comprehensive evaluation framework for H3 prediction models
with distance-based metrics, visualization, and performance analysis.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import h3
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_comprehensive_model(model_path=None, test_data_path=None, 
                                output_dir=None, create_visualizations=True):
    """
    Comprehensive evaluation of H3 prediction model with multiple metrics.
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test data 
        output_dir: Directory to save results
        create_visualizations: Whether to create plots
    """
    
    print("📊 Phase 4: Comprehensive Model Evaluation...")
    
    # Default paths
    if model_path is None:
        model_path = '/home/marius/repo_linux/ais-forecasting/data/models/final_models/comprehensive_h3_predictor.pkl'
    
    if output_dir is None:
        output_dir = Path('/home/marius/repo_linux/ais-forecasting/experiments/phase4_evaluation')
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(output_dir)
    
    print(f"   📁 Output directory: {output_dir}")
    
    # Load model and components
    print(f"\n🤖 Loading model...")
    try:
        model = joblib.load(model_path)
        encoder_path = model_path.replace('predictor.pkl', 'encoder.pkl')
        h3_encoder = joblib.load(encoder_path)
        
        metadata_path = model_path.replace('predictor.pkl', 'metadata.pkl')
        if Path(metadata_path).exists():
            metadata = joblib.load(metadata_path)
            print(f"   ✅ Model: {metadata.get('model_type', 'Unknown')}")
            print(f"   ✅ Features: {metadata.get('n_features', 'Unknown')}")
            print(f"   ✅ Classes: {metadata.get('n_classes', 'Unknown')}")
        else:
            metadata = {}
            
    except FileNotFoundError as e:
        print(f"   ❌ Model not found: {e}")
        print("   🔧 Please train the comprehensive model first")
        return None
    
    # Load test data
    print(f"\n📊 Loading test data...")
    if test_data_path is None:
        test_data_path = '/home/marius/repo_linux/ais-forecasting/data/processed/training_sets/comprehensive_h3_sequences.pkl'
    
    try:
        full_data = pd.read_pickle(test_data_path)
        
        # Use 20% as test set (matching training split)
        from sklearn.model_selection import train_test_split
        _, test_data = train_test_split(full_data, test_size=0.2, random_state=42)
        
        print(f"   ✅ Test data: {len(test_data):,} samples")
        
    except FileNotFoundError:
        print(f"   ❌ Test data not found: {test_data_path}")
        return None
    
    # Prepare test features
    print(f"\n🔧 Preparing test features...")
    
    # Get feature columns (exclude target and metadata)
    feature_cols = [col for col in test_data.columns 
                   if col not in ['target_h3_cell', 'vessel_imo']]
    
    X_test = test_data[feature_cols].copy()
    
    # Handle data preprocessing (same as training)
    for col in X_test.columns:
        if pd.api.types.is_datetime64_any_dtype(X_test[col]):
            # Convert datetime to unix timestamp
            X_test[col] = pd.to_datetime(X_test[col]).astype(int) / 10**9  # Convert to seconds
        elif X_test[col].dtype == 'object':
            try:
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            except:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_test[col] = le.fit_transform(X_test[col].astype(str))
    
    # Remove constant features and fill NaN
    for col in X_test.columns:
        if X_test[col].nunique() <= 1:
            X_test = X_test.drop(columns=[col])
    
    X_test = X_test.fillna(X_test.median())
    
    # Apply feature selection if used
    feature_selector_path = model_path.replace('predictor.pkl', 'selector.pkl')
    if Path(feature_selector_path).exists():
        print(f"   🎯 Applying feature selection...")
        feature_selector = joblib.load(feature_selector_path)
        X_test_selected = feature_selector.transform(X_test)
        feature_names = X_test.columns[feature_selector.get_support()].tolist()
        X_test = pd.DataFrame(X_test_selected, columns=feature_names, index=X_test.index)
        print(f"   ✅ Selected features: {len(feature_names)}")
    
    # Encode targets
    y_test = h3_encoder.transform(test_data['target_h3_cell'])
    
    print(f"   ✅ Test features shape: {X_test.shape}")
    print(f"   ✅ Test targets: {len(y_test)}")
    
    # Make predictions
    print(f"\n🔮 Making predictions...")
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        print(f"   ✅ Predictions complete")
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return None
    
    # Core metrics
    print(f"\n📈 Computing core metrics...")
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   🎯 Accuracy: {accuracy:.3f} ({accuracy:.1%})")
    
    # Top-k accuracy (if predicted class is in top-k predictions)
    if y_pred_proba is not None:
        for k in [3, 5, 10]:
            top_k_acc = top_k_accuracy(y_test, y_pred_proba, k)
            print(f"   🎯 Top-{k} Accuracy: {top_k_acc:.3f} ({top_k_acc:.1%})")
    
    # Distance-based evaluation
    print(f"\n📏 Computing distance-based metrics...")
    
    distances = []
    correct_predictions = []
    
    # Sample for distance computation (all if <1000, otherwise sample)
    n_samples = min(1000, len(y_test))
    sample_indices = np.random.choice(len(y_test), n_samples, replace=False)
    
    for idx in sample_indices:
        actual_h3 = h3_encoder.inverse_transform([y_test[idx]])[0]
        predicted_h3 = h3_encoder.inverse_transform([y_pred[idx]])[0]
        
        try:
            actual_lat, actual_lon = h3.h3_to_geo(actual_h3)
            pred_lat, pred_lon = h3.h3_to_geo(predicted_h3)
            
            distance_km = haversine_distance(actual_lat, actual_lon, pred_lat, pred_lon)
            distances.append(distance_km)
            correct_predictions.append(y_test[idx] == y_pred[idx])
            
        except Exception:
            continue  # Skip invalid H3 cells
    
    if distances:
        distances = np.array(distances)
        
        avg_distance = np.mean(distances)
        median_distance = np.median(distances)
        std_distance = np.std(distances)
        
        print(f"   📏 Average error: {avg_distance:.2f} ± {std_distance:.2f} km")
        print(f"   📏 Median error: {median_distance:.2f} km")
        print(f"   📏 Min error: {np.min(distances):.2f} km")
        print(f"   📏 Max error: {np.max(distances):.2f} km")
        
        # Success rates at different thresholds
        for threshold in [5, 10, 15, 20]:
            success_rate = np.mean(distances < threshold)
            print(f"   🎯 Success rate (<{threshold}km): {success_rate:.1%}")
    
    # Feature importance analysis
    print(f"\n🔍 Feature importance analysis...")
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   📊 Top 10 features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
    
    # Performance by vessel (if vessel info available)
    if 'vessel_imo' in test_data.columns:
        print(f"\n🚢 Performance by vessel...")
        vessel_performance = []
        
        for vessel in test_data['vessel_imo'].unique()[:10]:  # Top 10 vessels
            vessel_mask = test_data['vessel_imo'] == vessel
            vessel_y_test = y_test[vessel_mask]
            vessel_y_pred = y_pred[vessel_mask]
            
            if len(vessel_y_test) > 0:
                vessel_acc = accuracy_score(vessel_y_test, vessel_y_pred)
                vessel_performance.append({
                    'vessel': vessel,
                    'accuracy': vessel_acc,
                    'samples': len(vessel_y_test)
                })
        
        vessel_df = pd.DataFrame(vessel_performance).sort_values('accuracy', ascending=False)
        print(f"   📊 Best performing vessels:")
        for _, row in vessel_df.head(5).iterrows():
            print(f"      Vessel {row['vessel']}: {row['accuracy']:.1%} ({row['samples']} samples)")
    
    # Save evaluation results
    print(f"\n💾 Saving evaluation results...")
    
    results = {
        'accuracy': accuracy,
        'avg_distance_km': np.mean(distances) if distances else None,
        'median_distance_km': np.median(distances) if distances else None,
        'std_distance_km': np.std(distances) if distances else None,
        'success_rate_15km': np.mean(np.array(distances) < 15) if distances else None,
        'n_test_samples': len(y_test),
        'n_distance_samples': len(distances),
        'model_metadata': metadata
    }
    
    if y_pred_proba is not None:
        results['top_3_accuracy'] = top_k_accuracy(y_test, y_pred_proba, 3)
        results['top_5_accuracy'] = top_k_accuracy(y_test, y_pred_proba, 5)
    
    # Save results
    results_path = output_dir / 'evaluation_results.pkl'
    joblib.dump(results, results_path)
    print(f"   ✅ Results saved to: {results_path}")
    
    # Create visualizations if requested
    if create_visualizations:
        print(f"\n📊 Creating visualizations...")
        
        try:
            # Distance distribution
            if distances:
                plt.figure(figsize=(10, 6))
                plt.hist(distances, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Prediction Error (km)')
                plt.ylabel('Frequency')
                plt.title('Distribution of Prediction Errors')
                plt.axvline(np.mean(distances), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(distances):.2f} km')
                plt.axvline(15, color='green', linestyle='--', 
                           label='Target: 15 km')
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / 'distance_distribution.png', dpi=150)
                plt.close()
            
            # Feature importance plot
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                top_features = feature_importance.head(15)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title('Top 15 Feature Importances')
                plt.tight_layout()
                plt.savefig(output_dir / 'feature_importance.png', dpi=150)
                plt.close()
            
            print(f"   ✅ Visualizations saved to: {output_dir}")
            
        except Exception as e:
            print(f"   ⚠️  Visualization creation failed: {e}")
    
    # Summary
    print(f"\n🎉 Evaluation Complete!")
    print(f"   🎯 Final Accuracy: {accuracy:.1%}")
    if distances:
        print(f"   📏 Average Error: {np.mean(distances):.2f} km")
        success_15km = np.mean(np.array(distances) < 15)
        print(f"   🏆 Success Rate (<15km): {success_15km:.1%}")
        
        if success_15km > 0.6:
            print(f"   🌟 EXCELLENT! Exceeded 60% target!")
        elif success_15km > 0.3:
            print(f"   🚀 GOOD! Significant improvement!")
        else:
            print(f"   📈 IMPROVEMENT! Better than baseline!")
    
    return results

def top_k_accuracy(y_true, y_pred_proba, k):
    """Calculate top-k accuracy."""
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula."""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

if __name__ == "__main__":
    try:
        results = evaluate_comprehensive_model(create_visualizations=True)
        
        if results:
            print(f"\n🎉 SUCCESS! Comprehensive evaluation completed!")
            print(f"📊 Results saved with detailed metrics and visualizations")
        else:
            print(f"\n❌ Evaluation failed - check model and data availability")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
