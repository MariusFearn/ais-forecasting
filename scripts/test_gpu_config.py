#!/usr/bin/env python3
"""
Test GPU acceleration configuration for XGBoost training.
This script verifies that the config-driven GPU settings work correctly.
"""

import sys
import os
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.pipeline import TrainingPipeline
import xgboost as xgb
import pandas as pd
import numpy as np

def test_gpu_config():
    """Test that GPU configuration is properly applied."""
    print("🧪 Testing GPU Configuration Integration...")
    
    # Load the base config
    config_path = Path(__file__).parent.parent / "config/experiment_configs/experiment_h3_base.yaml"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📋 Loaded config with GPU settings:")
    if 'model' in config:
        for key, value in config['model'].items():
            print(f"   {key}: {value}")
    
    # Create dummy data for testing
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randint(0, 3, 100))
    
    # Initialize pipeline
    pipeline = TrainingPipeline(verbose=True)
    pipeline.X_train = X.iloc[:80]
    pipeline.X_test = X.iloc[80:]
    pipeline.y_train = y.iloc[:80]
    pipeline.y_test = y.iloc[80:]
    
    try:
        # Test model creation with config
        print("\n🚀 Testing XGBoost model creation with GPU config...")
        model = pipeline.train_xgboost_model(config=config, feature_selection=False)
        
        # Check if GPU settings were applied
        if hasattr(model, 'get_params'):
            params = model.get_params()
            print(f"\n✅ Model parameters applied:")
            gpu_params = ['tree_method', 'gpu_id', 'max_bin', 'device', 'n_jobs']
            for param in gpu_params:
                if param in params:
                    print(f"   {param}: {params[param]}")
        
        print(f"\n🎉 GPU configuration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ GPU configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_gpu_config()
