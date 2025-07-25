# AIS Vessel Trajectory Prediction - Quick Reference

## 🎯 Project Goal
Predict which H3 cell a vessel will visit next using machine learning.

## ✅ Current Working Status
- ✅ Phase 1-2 Complete: 65 vessel features extracted
- ✅ Working ML model: Random Forest classifier
- 🎯 Current accuracy: 5% (target: >60%)

## 🚀 Quick Commands

```bash
# Test feature extraction (100 records)
python scripts/test_simple.py

# Create training data (199 sequences) 
python scripts/create_simple_training_data.py

# Train Random Forest model
python scripts/train_simple_model.py
```

## 📁 Key Files
- `data/raw/` - AIS data (2018-2025)
- `src/features/vessel_h3_tracker.py` - GPS → H3 conversion
- `src/features/vessel_features.py` - 65 feature extraction
- `data/processed/training_sets/` - ML training data
- `data/models/final_models/` - Trained models

## 🎯 Next Steps
1. Scale up training data (more vessels)
2. Try XGBoost model
3. Add visualization tools
