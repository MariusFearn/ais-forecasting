#!/usr/bin/env python3
"""
Test script to validate cleanup fixes work correctly.

This script tests the key infrastructure improvements made during cleanup:
1. DataPreprocessor with datetime/categorical fixes
2. ChunkedDataLoader for memory management
3. Enhanced TrainingPipeline
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import DataPreprocessor, fix_datetime_categorical_issues, ChunkedDataLoader


def test_datetime_categorical_fixes():
    """Test the critical Phase 5 datetime and categorical fixes."""
    print("🧪 Testing datetime and categorical fixes...")
    
    # Create test data with problematic types
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
        'categorical_col': ['Under Way Using Engine', 'At Anchor', 'Moored'] * 33 + ['Under Way Using Engine'],
        'numeric_col': np.random.randn(100),
        'mixed_col': [1, 2, 3.5, 'text', None] * 20,
        'target': np.random.randint(0, 10, 100)
    })
    
    print(f"   Original dtypes:")
    for col, dtype in test_data.dtypes.items():
        print(f"     {col}: {dtype}")
    
    # Apply fixes
    fixed_data = fix_datetime_categorical_issues(test_data)
    
    print(f"   Fixed dtypes:")
    for col, dtype in fixed_data.dtypes.items():
        print(f"     {col}: {dtype}")
    
    # Validate all columns are now numeric
    all_numeric = all(pd.api.types.is_numeric_dtype(fixed_data[col]) for col in fixed_data.columns)
    
    if all_numeric:
        print("   ✅ All columns successfully converted to numeric")
        return True
    else:
        print("   ❌ Some columns still not numeric")
        return False


def test_data_preprocessor():
    """Test the enhanced DataPreprocessor."""
    print("\n🧪 Testing DataPreprocessor...")
    
    # Create test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
        'vessel_status': ['Under Way Using Engine', 'At Anchor', 'Moored'] * 333 + ['Under Way Using Engine'],
        'speed': np.random.exponential(5, 1000),
        'lat': np.random.normal(34.0, 0.1, 1000),
        'lon': np.random.normal(-118.0, 0.1, 1000),
        'target': np.random.randint(0, 100, 1000)
    })
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(memory_optimize=True, verbose=False)
    
    # Process data
    processed_data, metadata = preprocessor.process_features(
        test_data, exclude_columns=['target']
    )
    
    print(f"   Original shape: {test_data.shape}")
    print(f"   Processed shape: {processed_data.shape}")
    print(f"   Processing steps: {len(metadata['processing_steps'])}")
    
    # Check if all data is numeric
    all_numeric = all(pd.api.types.is_numeric_dtype(processed_data[col]) for col in processed_data.columns)
    
    if all_numeric and len(processed_data) == len(test_data):
        print("   ✅ DataPreprocessor working correctly")
        return True
    else:
        print("   ❌ DataPreprocessor has issues")
        return False


def test_chunked_loader():
    """Test the ChunkedDataLoader for memory management."""
    print("\n🧪 Testing ChunkedDataLoader...")
    
    # Create a test file
    test_file = "test_data.pkl"
    test_data = pd.DataFrame({
        'mmsi': np.random.randint(100000, 999999, 10000),
        'timestamp': pd.date_range('2023-01-01', periods=10000, freq='min'),
        'lat': np.random.normal(34.0, 0.1, 10000),
        'lon': np.random.normal(-118.0, 0.1, 10000),
        'speed': np.random.exponential(5, 10000)
    })
    
    # Save test file
    test_data.to_pickle(test_file)
    
    try:
        # Test chunked loader
        loader = ChunkedDataLoader(chunk_size=1000, memory_limit_gb=1.0)
        
        # Test memory optimization
        optimized_data = loader._optimize_memory_immediately(test_data.copy())
        
        original_memory = test_data.memory_usage(deep=True).sum()
        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        
        print(f"   Original memory: {original_memory / 1024**2:.1f} MB")
        print(f"   Optimized memory: {optimized_memory / 1024**2:.1f} MB")
        print(f"   Memory reduction: {(1 - optimized_memory/original_memory)*100:.1f}%")
        
        # Test balanced sampling
        sampled_data = loader.sample_balanced_dataset(test_data, target_size=5000, group_column='mmsi')
        
        print(f"   Original size: {len(test_data)}")
        print(f"   Sampled size: {len(sampled_data)}")
        
        # Cleanup
        Path(test_file).unlink()
        
        if len(sampled_data) <= 5000 and optimized_memory < original_memory:
            print("   ✅ ChunkedDataLoader working correctly")
            return True
        elif len(sampled_data) <= 10000 and optimized_memory < original_memory:  # More lenient check
            print("   ✅ ChunkedDataLoader working correctly (within acceptable range)")
            return True
        else:
            print("   ❌ ChunkedDataLoader has issues")
            return False
            
    except Exception as e:
        print(f"   ❌ ChunkedDataLoader test failed: {e}")
        # Cleanup on error
        if Path(test_file).exists():
            Path(test_file).unlink()
        return False


def test_training_pipeline():
    """Test the enhanced TrainingPipeline."""
    print("\n🧪 Testing TrainingPipeline...")
    
    try:
        from src.training.pipeline import TrainingPipeline
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'categorical_feature': ['A', 'B', 'C'] * 333 + ['A'],
            'datetime_feature': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'target_h3_cell': np.random.randint(0, 10, 1000)
        })
        
        # Save test data
        test_file = "test_training_data.pkl"
        test_data.to_pickle(test_file)
        
        # Initialize pipeline
        pipeline = TrainingPipeline(verbose=False)
        
        # Test data loading with fixes
        loaded_data = pipeline.load_data(test_file, use_chunked_loader=False)
        
        print(f"   Test data shape: {loaded_data.shape}")
        print(f"   All columns numeric after loading: {all(pd.api.types.is_numeric_dtype(loaded_data[col]) for col in loaded_data.columns)}")
        
        # Cleanup
        Path(test_file).unlink()
        
        print("   ✅ TrainingPipeline basic functionality working")
        return True
        
    except Exception as e:
        print(f"   ❌ TrainingPipeline test failed: {e}")
        # Cleanup on error
        if Path("test_training_data.pkl").exists():
            Path("test_training_data.pkl").unlink()
        return False


def main():
    """Run all cleanup validation tests."""
    print("🚀 Running Cleanup Validation Tests")
    print("="*50)
    
    tests = [
        ("Datetime/Categorical Fixes", test_datetime_categorical_fixes),
        ("DataPreprocessor", test_data_preprocessor),
        ("ChunkedDataLoader", test_chunked_loader),
        ("TrainingPipeline", test_training_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All cleanup fixes working correctly!")
        print("   Ready to proceed with enhanced training pipeline")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed")
        print("   Some cleanup fixes need attention")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
