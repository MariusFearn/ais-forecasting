# AIS Vessel Trajectory Prediction

A simple machine learning system for predicting individual vessel movements using AIS data and H3 geospatial indexing.

## 🎯 Simple Goal
**Predict which H3 cell a vessel will visit next** based on its current position and movement patterns.

## ✅ Current Status

### COMPLETED (Phase 1-2):
- **Data**: 8 years Cape Town AIS data (2018-2025, 8.3M+ records)
- **H3 Indexing**: Resolution 5 (8.54km edge length) 
- **Vessel Tracking**: Convert GPS → H3 cell sequences
- **Feature Engineering**: 65 vessel features extracted and validated
- **Working ML Pipeline**: Random Forest classifier trained (91.8% train accuracy)

### 🎯 CURRENT FOCUS (Phase 3):
- ✅ Created training data (199 sequences)
- ✅ Trained first ML model (Random Forest)
- 🎯 Improve accuracy with more data/better models
- 🎯 Add visualization and evaluation tools

## 🚀 Quick Start

### 1. Test the Working Model
```python
# Run our current working example
python scripts/test_simple.py                    # Test feature extraction
python scripts/create_simple_training_data.py   # Create training data  
python scripts/train_simple_model.py            # Train ML model
```

### 2. Project Structure
```
ais-forecasting/
├── data/
│   ├── raw/                    # AIS data files (2018-2025)
│   ├── processed/              
│   │   ├── training_sets/      # ✅ ML training data
│   │   └── vessel_features/    # ✅ 65 features per vessel
│   └── models/                 # ✅ Trained models
├── src/
│   ├── features/               # ✅ COMPLETE
│   │   ├── vessel_h3_tracker.py   # GPS → H3 sequences
│   │   └── vessel_features.py     # 65 feature extraction
│   └── models/                 # 🎯 Simple classifiers
├── scripts/                    # ✅ Working examples
└── notebooks/
    └── vessel_exploration.ipynb   # ✅ Complete analysis
```

## 📊 Data Summary

### AIS Data Files (data/raw/):
- **8 files**: `ais_cape_data_2018.pkl` to `ais_cape_data_2025.pkl`
- **Format**: Pandas DataFrames with 18 columns
- **Sample size**: 8.3M+ records (2018), ~1.1GB per year
- **Coverage**: Cape Town maritime area, UTC timestamps
- **Key columns**: `imo` (vessel ID), `lat/lon` (position), `speed`, `heading`, `mdt` (timestamp)

### Feature Engineering (65 features):
- **Current State**: Position, speed, heading, time in cell
- **Movement History**: 6h/12h/24h speed/direction patterns  
- **Journey Context**: Distance traveled, cells visited, journey phase
- **Operational Context**: Cargo status, port proximity, coastal distance

## 🤖 Machine Learning Pipeline

### Current Working Example:
```
Raw AIS Data → H3 Sequences → 65 Features → Random Forest → Next Cell Prediction
```

### Model Performance:
- **Training Accuracy**: 91.8% (shows model learns patterns)
- **Test Accuracy**: 5.0% (162 possible cells, room for improvement)
- **Feature Importance**: Location (lat/lon) most important, then current H3 cell

### Success Criteria:
- **Target**: >60% accuracy predicting next H3 cell
- **Distance**: <15km average error from actual position

## 🔧 Technical Implementation

### Phase 2 Accomplishments:
- **VesselH3Tracker**: Converts GPS data to H3 sequences with quality validation
- **VesselFeatureExtractor**: Creates 65 comprehensive vessel features
- **Tested**: Validated on vessel IMO 9883089 (364-day journey, 9,151 records, 1,530 H3 cells)

### Next Steps:
1. **More training data** - Add more vessels to training set
2. **Better features** - Use more of the 65 available features
3. **Advanced models** - Try XGBoost, Neural Networks
4. **Evaluation tools** - Visualize predictions on maps

## 📦 Dependencies

**Core**: pandas, numpy, scikit-learn, h3-py  
**Geospatial**: geopandas, folium  
**ML**: pytorch, optuna (for advanced models)  
**Analysis**: jupyter, matplotlib, seaborn

Install: `pip install -r requirements.txt`

## 🎯 Why This Approach Works

1. **Clear Success Metrics** - Easy to measure if predictions are correct
2. **Solid Foundation** - 65 validated features from real vessel data
3. **Simple First** - Random Forest before complex deep learning
4. **Extensible** - Can add multi-step prediction, fleet patterns later
5. **Real Value** - Vessel operators want to know where ships go next

---

**Focus**: Simple vessel next-cell prediction that actually works → then extend to more complex features.

1. **Create training data** - Convert 65 features to input-target pairs
2. **Train classifier** - Start with Random Forest
3. **Evaluate accuracy** - Classification metrics + distance errors
4. **Visualize results** - Show predicted vs actual paths
5. **Iterate** - Improve features and try different models

**Goal**: Get our first working ML model predicting vessel movements!

## Contact

For questions about this project, open an issue or contact the maintainer.

---

*Focus: Simple vessel trajectory prediction that works, then extend later.*
