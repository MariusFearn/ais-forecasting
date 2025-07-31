#!/usr/bin/env python3
"""
Unified model evaluation system for AIS forecasting models.
Supports multiple evaluation types and model formats.
"""

import argparse
import sys
import yaml
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    import h3
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


class UnifiedModelEvaluator:
    """Unified evaluation system for all AIS forecasting models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator with configuration."""
        self.config = config
        self.eval_config = config['evaluation']
        self.eval_type = self.eval_config['type']
        self.results = {}
        
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation based on configuration type."""
        print(f"📊 Running {self.eval_config['name']} ({self.eval_type})")
        print("=" * 60)
        
        if self.eval_type == 'simple':
            return self._simple_evaluation()
        elif self.eval_type == 'comprehensive':
            return self._comprehensive_evaluation()
        elif self.eval_type == 'production':
            return self._production_evaluation()
        elif self.eval_type == 'comparative':
            return self._comparative_evaluation()
        else:
            raise ValueError(f"Unknown evaluation type: {self.eval_type}")
    
    def _load_model_components(self, model_config: Dict[str, Any]) -> tuple:
        """Load model, encoder, and metadata."""
        if model_config.get('auto_detect_paths', False):
            # Auto-detect latest model paths
            model_dir = Path("data/models/final_models")
            model_path = model_dir / "comprehensive_h3_predictor.pkl"
            encoder_path = model_dir / "comprehensive_h3_encoder.pkl"
            metadata_path = model_dir / "comprehensive_model_metadata.pkl"
        else:
            # Use specified paths
            model_path = Path(model_config['model_path'])
            encoder_path = Path(model_config['encoder_path'])
            metadata_path = Path(model_config['metadata_path'])
        
        print(f"🤖 Loading model components...")
        print(f"   Model: {model_path}")
        
        try:
            model = joblib.load(model_path)
            h3_encoder = joblib.load(encoder_path)
            metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
            
            print(f"   ✅ Model loaded: {metadata.get('model_type', 'Unknown')}")
            print(f"   ✅ Features: {metadata.get('n_features', 'Unknown')}")
            
            return model, h3_encoder, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model components: {e}")
    
    def _load_test_data(self, data_config: Dict[str, Any]) -> pd.DataFrame:
        """Load and prepare test data."""
        if 'test_data_path' in data_config:
            # Load from specific test file
            test_data_path = data_config['test_data_path']
            print(f"📊 Loading test data from: {test_data_path}")
            return pd.read_pickle(test_data_path)
        else:
            # Load from training data and split
            training_data_path = data_config.get('training_data_path', 
                'data/processed/training_sets/comprehensive_h3_sequences.pkl')
            print(f"📊 Loading training data from: {training_data_path}")
            
            full_data = pd.read_pickle(training_data_path)
            
            # Split to get test set (using same random state as training)
            _, test_data = train_test_split(full_data, test_size=0.2, random_state=42)
            
            print(f"   ✅ Test set size: {len(test_data)} samples")
            return test_data
    
    def _simple_evaluation(self) -> Dict[str, Any]:
        """Basic model evaluation with core metrics."""
        print("📈 Running Simple Evaluation...")
        
        try:
            # Load model and data
            model, h3_encoder, metadata = self._load_model_components(self.config['model'])
            test_data = self._load_test_data(self.config.get('data', {}))
            
            # Prepare features and targets
            feature_cols = [col for col in test_data.columns 
                          if col not in ['target_h3_cell', 'vessel_imo']]
            
            X_test = test_data[feature_cols].copy()
            y_test = h3_encoder.transform(test_data['target_h3_cell'])
            
            # Clean data
            X_test = self._clean_features(X_test)
            
            # Apply feature selection if exists
            X_test = self._apply_feature_selection(X_test)
            
            # Make predictions
            print(f"\n🔮 Making predictions...")
            y_pred = model.predict(X_test)
            
            # Core metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n📊 Simple Evaluation Results:")
            print(f"   🎯 Accuracy: {accuracy:.3f} ({accuracy:.1%})")
            print(f"   📈 Test samples: {len(y_test):,}")
            
            return {
                'status': 'success',
                'accuracy': accuracy,
                'n_test_samples': len(y_test),
                'model_metadata': metadata
            }
            
        except Exception as e:
            print(f"❌ Simple evaluation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _comprehensive_evaluation(self) -> Dict[str, Any]:
        """Detailed evaluation with visualizations and analysis."""
        print("🔍 Running Comprehensive Evaluation...")
        
        try:
            # Load model and data
            model, h3_encoder, metadata = self._load_model_components(self.config['model'])
            test_data = self._load_test_data(self.config.get('data', {}))
            
            # Prepare features and targets
            feature_cols = [col for col in test_data.columns 
                          if col not in ['target_h3_cell', 'vessel_imo']]
            
            X_test = test_data[feature_cols].copy()
            y_test = h3_encoder.transform(test_data['target_h3_cell'])
            
            # Clean data
            X_test = self._clean_features(X_test)
            
            # Apply feature selection if exists
            X_test = self._apply_feature_selection(X_test)
            
            print(f"   ✅ Test features shape: {X_test.shape}")
            print(f"   ✅ Test targets: {len(y_test)}")
            
            # Make predictions
            print(f"\n🔮 Making predictions...")
            y_pred = model.predict(X_test)
            
            # Try to get prediction probabilities
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass
            
            # Core metrics
            accuracy = accuracy_score(y_test, y_pred)
            print(f"   🎯 Accuracy: {accuracy:.3f} ({accuracy:.1%})")
            
            # Top-k accuracy if probabilities available
            top_k_scores = {}
            if y_pred_proba is not None:
                for k in [3, 5]:
                    top_k_acc = self._top_k_accuracy(y_test, y_pred_proba, k)
                    top_k_scores[f'top_{k}'] = top_k_acc
                    print(f"   📊 Top-{k} accuracy: {top_k_acc:.3f} ({top_k_acc:.1%})")
            
            # Distance-based evaluation
            distance_results = {}
            if self.config.get('analysis', {}).get('distance_analysis', {}).get('enable', False):
                distance_results = self._distance_based_evaluation(
                    y_test, y_pred, h3_encoder, 
                    self.config['analysis']['distance_analysis']
                )
            
            # Feature importance analysis
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = X_test.columns.tolist()
                importances = model.feature_importances_
                
                feature_importance = dict(zip(feature_names, importances))
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n🏆 Top 5 Most Important Features:")
                for i, (feature, importance) in enumerate(sorted_features[:5]):
                    print(f"   {i+1}. {feature}: {importance:.3f}")
            
            # Performance by vessel (if vessel info available)
            vessel_performance = {}
            if 'vessel_imo' in test_data.columns:
                vessel_performance = self._performance_by_vessel(
                    test_data, y_test, y_pred, h3_encoder
                )
            
            # Create visualizations if requested
            if self.config.get('analysis', {}).get('visualization', {}).get('create_plots', False):
                self._create_visualizations(y_test, y_pred, feature_importance, distance_results)
            
            print(f"\n🎉 Comprehensive Evaluation Complete!")
            print(f"   🎯 Final Accuracy: {accuracy:.1%}")
            if distance_results:
                avg_distance = distance_results.get('avg_distance_km', 0)
                success_rate = distance_results.get('success_rate_15km', 0)
                print(f"   📏 Average distance error: {avg_distance:.1f}km")
                print(f"   🎯 Success rate (15km): {success_rate:.1%}")
            
            return {
                'status': 'success',
                'accuracy': accuracy,
                'top_k_accuracy': top_k_scores,
                'distance_results': distance_results,
                'feature_importance': feature_importance,
                'vessel_performance': vessel_performance,
                'n_test_samples': len(y_test),
                'model_metadata': metadata
            }
            
        except Exception as e:
            print(f"❌ Comprehensive evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def _production_evaluation(self) -> Dict[str, Any]:
        """Production-ready evaluation with deployment metrics."""
        print("🚀 Running Production Evaluation...")
        
        # This would include deployment-specific metrics
        # For now, run comprehensive evaluation with production focus
        results = self._comprehensive_evaluation()
        
        if results['status'] == 'success':
            # Add production-specific checks
            accuracy = results['accuracy']
            
            production_ready = accuracy >= 0.8  # 80% threshold for production
            
            print(f"\n🏭 Production Readiness Assessment:")
            print(f"   Accuracy threshold (80%): {'✅ PASSED' if production_ready else '❌ FAILED'}")
            
            results['production_ready'] = production_ready
            results['production_threshold'] = 0.8
        
        return results
    
    def _comparative_evaluation(self) -> Dict[str, Any]:
        """Compare multiple models."""
        print("🔄 Running Comparative Evaluation...")
        
        # This would load and compare multiple models
        # For now, provide framework for future implementation
        print("   Comparative evaluation framework ready")
        print("   Would compare multiple model types (XGBoost, RandomForest, etc.)")
        
        return {
            'status': 'success',
            'message': 'Comparative evaluation framework - implementation pending'
        }
    
    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for evaluation."""
        # Handle non-numeric columns
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                # Try to convert to numeric
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Remove constant features
        for col in X.columns:
            if X[col].nunique() <= 1:
                X = X.drop(columns=[col])
        
        # Fill NaN values
        X = X.fillna(X.median())
        
        return X
    
    def _apply_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection if selector exists."""
        selector_path = Path("data/models/final_models/comprehensive_h3_selector.pkl")
        
        if selector_path.exists():
            try:
                feature_selector = joblib.load(selector_path)
                X = pd.DataFrame(
                    feature_selector.transform(X),
                    columns=X.columns[feature_selector.get_support()],
                    index=X.index
                )
                print(f"   ✅ Feature selection applied: {X.shape[1]} features selected")
            except Exception as e:
                print(f"   ⚠️  Feature selection failed: {e}")
        
        return X
    
    def _top_k_accuracy(self, y_true: np.ndarray, y_pred_proba: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy."""
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    
    def _distance_based_evaluation(self, y_test: np.ndarray, y_pred: np.ndarray, 
                                 h3_encoder, config: Dict[str, Any]) -> Dict[str, float]:
        """Maritime-specific distance-based metrics."""
        print(f"\n📏 Computing distance-based metrics...")
        
        distances = []
        sample_size = min(config.get('sample_size', 1000), len(y_test))
        sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
        
        for idx in sample_indices:
            true_cell = h3_encoder.inverse_transform([y_test[idx]])[0]
            pred_cell = h3_encoder.inverse_transform([y_pred[idx]])[0]
            
            # Get cell centers
            true_lat, true_lon = h3.h3_to_geo(true_cell)
            pred_lat, pred_lon = h3.h3_to_geo(pred_cell)
            
            # Calculate distance
            distance = self._haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
            distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            median_distance = np.median(distances)
            
            # Success rates at different thresholds
            success_rates = {}
            for threshold in config.get('target_distances', [5, 10, 15, 20]):
                success_rate = np.mean(np.array(distances) < threshold)
                success_rates[f'success_rate_{threshold}km'] = success_rate
                print(f"   {threshold}km success rate: {success_rate:.1%}")
            
            print(f"   Average distance: {avg_distance:.1f}km")
            print(f"   Median distance: {median_distance:.1f}km")
            
            return {
                'avg_distance_km': avg_distance,
                'median_distance_km': median_distance,
                'n_distance_samples': len(distances),
                **success_rates
            }
        
        return {}
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _performance_by_vessel(self, test_data: pd.DataFrame, y_test: np.ndarray, 
                             y_pred: np.ndarray, h3_encoder) -> Dict[str, float]:
        """Analyze performance by individual vessels."""
        print(f"\n🚢 Analyzing performance by vessel...")
        
        vessel_performance = {}
        
        for vessel_imo in test_data['vessel_imo'].unique()[:5]:  # Top 5 vessels
            vessel_mask = test_data['vessel_imo'] == vessel_imo
            vessel_y_test = y_test[vessel_mask]
            vessel_y_pred = y_pred[vessel_mask]
            
            if len(vessel_y_test) > 10:  # Only analyze vessels with sufficient data
                vessel_accuracy = accuracy_score(vessel_y_test, vessel_y_pred)
                vessel_performance[str(vessel_imo)] = vessel_accuracy
                print(f"   Vessel {vessel_imo}: {vessel_accuracy:.1%} ({len(vessel_y_test)} samples)")
        
        return vessel_performance
    
    def _create_visualizations(self, y_test: np.ndarray, y_pred: np.ndarray, 
                             feature_importance: Dict[str, float], 
                             distance_results: Dict[str, float]):
        """Create evaluation visualizations."""
        print(f"\n📊 Creating visualizations...")
        
        output_dir = Path("experiments/evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix (sample if too large)
        if len(np.unique(y_test)) <= 50:  # Only for manageable number of classes
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Confusion matrix saved")
        
        # Feature importance
        if feature_importance:
            plt.figure(figsize=(10, 6))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:15]  # Top 15 features
            
            features, importances = zip(*top_features)
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Most Important Features')
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Feature importance plot saved")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def list_available_configs():
    """List available evaluation configurations."""
    config_dir = Path(__file__).parent.parent / "config" / "experiment_configs"
    eval_configs = list(config_dir.glob("*evaluation*.yaml"))
    
    print("📋 Available evaluation configurations:")
    for config_file in sorted(eval_configs):
        config_name = config_file.stem
        print(f"   • {config_name}")
    
    if not eval_configs:
        print("   No evaluation configurations found in config/experiment_configs/")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Unified model evaluation system")
    parser.add_argument("--config", "-c", help="Evaluation configuration name (without .yaml)")
    parser.add_argument("--list-configs", action="store_true", help="List available evaluation configurations")
    parser.add_argument("--model-path", help="Override model path from config")
    
    args = parser.parse_args()
    
    if args.list_configs:
        list_available_configs()
        return
    
    if not args.config:
        print("❌ Please specify an evaluation configuration with --config")
        print("Use --list-configs to see available configurations")
        return 1
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "experiment_configs" / f"{args.config}.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Use --list-configs to see available configurations")
        return 1
    
    try:
        config = load_config(config_path)
        
        # Override model path if provided
        if args.model_path:
            config['model']['model_path'] = args.model_path
        
        # Create and run evaluator
        evaluator = UnifiedModelEvaluator(config)
        results = evaluator.evaluate()
        
        # Save results
        output_dir = Path("experiments/evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f"{args.config}_results.pkl"
        joblib.dump(results, results_path)
        print(f"\n💾 Results saved to: {results_path}")
        
        # Exit with appropriate code
        if results.get('status') == 'success':
            print(f"\n🎉 Evaluation completed successfully!")
            return 0
        else:
            print(f"\n❌ Evaluation failed. Check output above for details.")
            return 1
            
    except Exception as e:
        print(f"❌ Evaluation system failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
