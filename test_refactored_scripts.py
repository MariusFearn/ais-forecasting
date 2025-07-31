#!/usr/bin/env python3
"""
Quick test script to verify refactored functionality works.
"""

import sys
import os
from pathlib import Path
import pickle
import pandas as pd

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_simple_model_prediction():
    """Test that we can load and use the simple model for prediction"""
    try:
        # Load the trained model
        model_path = project_root / "data/models/final_models/simple_h3_predictor.pkl"
        encoder_path = project_root / "data/models/final_models/h3_label_encoder.pkl"
        
        if not model_path.exists():
            print("❌ Simple model not found. Run train_simple_model.py first.")
            return False
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
            
        # Test data
        test_features = [
            ['854a71a7fffffff', 10.0, 90.0, -33.9, 18.4, 300.0]  # sample input
        ]
        feature_names = ['current_h3_cell', 'current_speed', 'current_heading', 'lat', 'lon', 'time_in_current_cell']
        
        # Create test dataframe  
        test_df = pd.DataFrame(test_features, columns=feature_names)
        
        # Encode H3 cell
        test_df['current_h3_encoded'] = encoder.transform(test_df['current_h3_cell'])
        
        # Prepare features for prediction
        X = test_df[['current_h3_encoded', 'current_speed', 'current_heading', 'lat', 'lon', 'time_in_current_cell']]
        
        # Make prediction
        prediction = model.predict(X)
        predicted_h3 = encoder.inverse_transform(prediction)[0]
        
        print(f"✅ Simple model prediction test successful!")
        print(f"   Input: H3={test_features[0][0]}, Speed={test_features[0][1]}, Heading={test_features[0][2]}")
        print(f"   Predicted next H3: {predicted_h3}")
        return True
        
    except Exception as e:
        print(f"❌ Simple model prediction test failed: {e}")
        return False

def test_enhanced_model_prediction():
    """Test that we can load and use the enhanced model for prediction"""
    try:
        # Load the trained model
        model_path = project_root / "data/models/final_models/enhanced_h3_predictor.pkl"
        h3_encoder_path = project_root / "data/models/final_models/enhanced_h3_label_encoder.pkl"
        vessel_encoder_path = project_root / "data/models/final_models/vessel_label_encoder.pkl"
        
        if not model_path.exists():
            print("❌ Enhanced model not found. Run train_enhanced_model.py first.")
            return False
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        with open(h3_encoder_path, 'rb') as f:
            h3_encoder = pickle.load(f)
            
        with open(vessel_encoder_path, 'rb') as f:
            vessel_encoder = pickle.load(f)
            
        # Test data - use a vessel from training
        test_features = [
            [9883089, '854a71a7fffffff', 8.5, 95.0, -33.9, 18.4, 250.0]  # sample input
        ]
        feature_names = ['vessel_imo', 'current_h3_cell', 'current_speed', 'current_heading', 'lat', 'lon', 'time_in_current_cell']
        
        # Create test dataframe  
        test_df = pd.DataFrame(test_features, columns=feature_names)
        
        # Encode features
        test_df['vessel_encoded'] = vessel_encoder.transform(test_df['vessel_imo'])
        test_df['current_h3_encoded'] = h3_encoder.transform(test_df['current_h3_cell'])
        
        # Add engineered features
        test_df['lat_lon_interaction'] = test_df['lat'] * test_df['lon']
        test_df['speed_heading_interaction'] = test_df['current_speed'] * test_df['current_heading']
        
        # Prepare features for prediction
        X = test_df[['vessel_encoded', 'current_h3_encoded', 'current_speed', 'current_heading', 
                     'lat', 'lon', 'time_in_current_cell', 'lat_lon_interaction', 'speed_heading_interaction']]
        
        # Make prediction
        prediction = model.predict(X)
        predicted_h3 = h3_encoder.inverse_transform(prediction)[0]
        
        print(f"✅ Enhanced model prediction test successful!")
        print(f"   Input: Vessel={test_features[0][0]}, H3={test_features[0][1]}, Speed={test_features[0][2]}")
        print(f"   Predicted next H3: {predicted_h3}")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced model prediction test failed: {e}")
        return False

def test_data_loading():
    """Test that we can load the training data"""
    try:
        # Test simple data
        simple_data_path = project_root / "data/processed/training_sets/vessel_h3_sequences.pkl"
        if simple_data_path.exists():
            with open(simple_data_path, 'rb') as f:
                simple_data = pickle.load(f)
            print(f"✅ Simple training data loaded: {len(simple_data)} samples")
        else:
            print("❌ Simple training data not found")
            
        # Test multi-vessel data
        multi_data_path = project_root / "data/processed/training_sets/multi_vessel_h3_sequences.pkl"
        if multi_data_path.exists():
            with open(multi_data_path, 'rb') as f:
                multi_data = pickle.load(f)
            print(f"✅ Multi-vessel training data loaded: {len(multi_data)} samples")
            return True
        else:
            print("❌ Multi-vessel training data not found")
            return False
            
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Refactored Scripts Functionality...")
    print("=" * 50)
    
    # Test data loading
    print("\n📊 Testing Data Loading...")
    data_ok = test_data_loading()
    
    # Test simple model prediction
    print("\n🤖 Testing Simple Model Prediction...")
    simple_ok = test_simple_model_prediction()
    
    # Test enhanced model prediction  
    print("\n🚀 Testing Enhanced Model Prediction...")
    enhanced_ok = test_enhanced_model_prediction()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"   Data Loading: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"   Simple Model: {'✅ PASS' if simple_ok else '❌ FAIL'}")
    print(f"   Enhanced Model: {'✅ PASS' if enhanced_ok else '❌ FAIL'}")
    
    if all([data_ok, simple_ok, enhanced_ok]):
        print("\n🎉 ALL TESTS PASSED! Refactoring successful!")
    else:
        print("\n⚠️  Some tests failed. Check outputs above.")
