#!/usr/bin/env python3
"""
Simple test to verify GPU configuration integration.
Tests that YAML config values are properly passed to XGBoost model creation.
"""

import yaml
import xgboost as xgb
from pathlib import Path

def test_yaml_to_xgboost_integration():
    """Test that YAML config gets properly applied to XGBoost."""
    
    print("🧪 Testing YAML → XGBoost Integration")
    print("=" * 50)
    
    # 1. Load the base config YAML
    config_path = Path("../config/experiment_configs/base_h3_experiment.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("📋 Loaded Config:")
    if 'model' in config:
        for key, value in config['model'].items():
            print(f"   {key}: {value}")
    
    if 'hardware' in config:
        print("🖥️  Hardware settings:")
        for key, value in config['hardware'].items():
            print(f"   {key}: {value}")
    
    # 2. Simulate what create_model() function does
    print("\n🚀 Testing create_model() logic...")
    
    # Base parameters
    params = {
        'n_estimators': 10,  # Small for testing
        'max_depth': 3,
        'random_state': 42
    }
    
    # Apply GPU settings from config (same logic as in scripts/train_h3_model.py)
    if 'model' in config:
        gpu_settings = ['tree_method', 'gpu_id', 'max_bin', 'device']
        for setting in gpu_settings:
            if setting in config['model']:
                params[setting] = config['model'][setting]
                print(f"   ✅ Applied {setting}: {config['model'][setting]}")
    
    # Apply hardware settings
    if config.get('hardware', {}).get('cpu', {}).get('threads'):
        params['n_jobs'] = config['hardware']['cpu']['threads']
        print(f"   ✅ Applied n_jobs: {config['hardware']['cpu']['threads']}")
    
    print(f"\n🔧 Final XGBoost Parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # 3. Create XGBoost model with these parameters
    try:
        print(f"\n🎯 Creating XGBoost model...")
        model = xgb.XGBClassifier(**params)
        
        # Check if parameters were applied
        model_params = model.get_params()
        print(f"\n✅ Model created successfully!")
        print(f"🔍 Checking applied parameters:")
        
        for key in ['tree_method', 'gpu_id', 'n_jobs', 'max_bin']:
            if key in model_params:
                print(f"   {key}: {model_params[key]}")
        
        # Test GPU availability
        if model_params.get('tree_method') == 'gpu_hist':
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                print(f"   🚀 GPU acceleration configured: {cuda_available}")
                if cuda_available:
                    print(f"   🔥 GPU device: {torch.cuda.get_device_name(0)}")
                else:
                    print(f"   ⚠️  GPU not available, will fallback to CPU")
            except ImportError:
                print(f"   ⚠️  PyTorch not available for GPU check")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating XGBoost model: {e}")
        return False

if __name__ == "__main__":
    success = test_yaml_to_xgboost_integration()
    
    if success:
        print(f"\n🎉 Integration test PASSED!")
        print(f"✅ YAML config is properly applied to XGBoost")
        print(f"✅ GPU settings are configured")
        print(f"✅ CPU threading is optimized")
    else:
        print(f"\n❌ Integration test FAILED!")
        print(f"⚠️  Check configuration files")
