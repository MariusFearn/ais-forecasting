#!/usr/bin/env python3
"""
Train a simple Random Forest model to predict next H3 cell
The simplest possible ML approach that should work
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def train_simple_h3_predictor():
    """Train the simplest possible H3 cell predictor"""
    
    print("🤖 Training Simple H3 Cell Predictor...")
    
    # Load the training data we just created
    print("\n📊 Loading training data...")
    training_df = pd.read_pickle('/home/marius/repo_linux/ais-forecasting/data/processed/training_sets/simple_h3_sequences.pkl')
    
    print(f"   ✅ Loaded {len(training_df)} training examples")
    print(f"   📊 Features: {list(training_df.columns[:-1])}")
    
    # Prepare features and target
    print("\n🔧 Preparing features...")
    
    # Encode H3 cells as numbers (ML models need numbers)
    h3_encoder = LabelEncoder()
    all_h3_cells = list(set(training_df['current_h3_cell'].tolist() + training_df['target_h3_cell'].tolist()))
    h3_encoder.fit(all_h3_cells)
    
    # Features (X)
    X = training_df[['current_speed', 'current_heading', 'lat', 'lon', 'time_in_current_cell']].copy()
    X['current_h3_encoded'] = h3_encoder.transform(training_df['current_h3_cell'])
    
    # Target (y)
    y = h3_encoder.transform(training_df['target_h3_cell'])
    
    print(f"   ✅ Features shape: {X.shape}")
    print(f"   ✅ Target shape: {y.shape}")
    print(f"   ✅ Unique H3 cells: {len(h3_encoder.classes_)}")
    
    # Split into train/test
    print("\n✂️  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   📊 Train: {len(X_train)} samples")
    print(f"   📊 Test: {len(X_test)} samples")
    
    # Train Random Forest (simple but effective)
    print("\n🌲 Training Random Forest...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    print("   ✅ Model trained!")
    
    # Evaluate
    print("\n📊 Evaluating model...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"   📈 Train Accuracy: {train_accuracy:.3f}")
    print(f"   📈 Test Accuracy: {test_accuracy:.3f}")
    
    # Feature importance
    print("\n🔍 Feature Importance:")
    feature_names = X.columns
    importances = model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"   {name}: {importance:.3f}")
    
    # Save the model
    print("\n💾 Saving model...")
    model_path = '/home/marius/repo_linux/ais-forecasting/data/models/final_models/simple_h3_predictor.pkl'
    encoder_path = '/home/marius/repo_linux/ais-forecasting/data/models/final_models/h3_label_encoder.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(h3_encoder, encoder_path)
    
    print(f"   ✅ Model saved to: {model_path}")
    print(f"   ✅ Encoder saved to: {encoder_path}")
    
    # Quick prediction test
    print("\n🔮 Testing prediction...")
    sample_input = X_test.iloc[0:1]
    predicted_encoded = model.predict(sample_input)[0]
    predicted_h3 = h3_encoder.inverse_transform([predicted_encoded])[0]
    actual_h3 = h3_encoder.inverse_transform([y_test[0]])[0]
    
    print(f"   📍 Sample prediction:")
    print(f"      Input: speed={sample_input['current_speed'].iloc[0]:.1f}, heading={sample_input['current_heading'].iloc[0]:.1f}")
    print(f"      Predicted H3: {predicted_h3}")
    print(f"      Actual H3: {actual_h3}")
    print(f"      Correct: {'✅' if predicted_h3 == actual_h3 else '❌'}")
    
    return model, h3_encoder, test_accuracy

if __name__ == "__main__":
    try:
        model, encoder, accuracy = train_simple_h3_predictor()
        print(f"\n🎉 SUCCESS! H3 Predictor trained with {accuracy:.1%} accuracy!")
        print("🚀 You now have a working ML model that predicts vessel movements!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
