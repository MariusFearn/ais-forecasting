# AIS Vessel Trajectory Prediction

A simple machine learning system for predicting individual vessel movements using AIS data and H3 geospatial indexing.

## Simple Goal
**Predict which H3 cell a vessel will visit next** based on its current position and movement patterns.

## Why This Approach?
- ✅ Clear input → output relationship  
- ✅ Easy to evaluate (classification accuracy)
- ✅ Builds on completed Phase 2 work (65 features)
- ✅ Manageable scope for first ML success

## Current Status

### ✅ COMPLETED (Phase 1-2):
- **Data**: 8 years Cape Town AIS data (2018-2025)
- **H3 Indexing**: Resolution 5 (8.54km edge length) 
- **Vessel Tracking**: Convert GPS → H3 cell sequences
- **Feature Engineering**: 65 vessel features extracted
- **Validation**: Tested on real vessel (364-day journey)

### 🎯 NEXT (Phase 3):
- Create training datasets (input features → target next cell)
- Train simple classifier (Random Forest/Gradient Boosting)
- Evaluate prediction accuracy
- Visualize results

## Project Structure

```
ais-forecasting/
├── data/
│   ├── raw/                    # AIS data files (2018-2025)
│   └── processed/              
│       ├── vessel_features/    # ✅ 65 features per vessel
│       ├── training_sets/      # 🎯 ML input-target pairs
│       └── predictions/        # 🎯 Model outputs
├── src/
│   ├── features/               # ✅ COMPLETE
│   │   ├── vessel_h3_tracker.py   # GPS → H3 sequences
│   │   └── vessel_features.py     # 65 feature extraction
│   ├── models/                 # 🎯 Simple classifiers
│   └── data/                   # Data loading utilities
├── notebooks/
│   └── vessel_exploration.ipynb   # ✅ Complete analysis
└── scripts/                    # 🎯 Training/evaluation scripts
```

## Quick Start

### 1. Environment Setup
```bash
git clone <repository>
cd ais-forecasting
pip install -r requirements.txt
```

### 2. Simple Prediction Example
```python
# Load and process vessel data
from src.features.vessel_h3_tracker import VesselH3Tracker
from src.features.vessel_features import VesselFeatureExtractor

# Convert vessel to H3 sequence
tracker = VesselH3Tracker(resolution=5)
h3_sequence = tracker.convert_vessel_to_h3_sequence(vessel_data)

# Extract 65 features
extractor = VesselFeatureExtractor()
features = extractor.extract_features(h3_sequence)

# Train simple model (coming in Phase 3)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_features, y_next_cell)

# Predict next cell
predicted_cell = model.predict(current_features)
```

## Technical Approach

### Input: 65 Vessel Features
- **Current State**: Position, speed, heading, time in cell
- **Movement History**: 6h/12h/24h speed/direction patterns  
- **Journey Context**: Distance traveled, cells visited, journey phase
- **Operational Context**: Cargo status, port proximity, coastal distance

### Output: Next H3 Cell
- Classification problem: Which of ~1500 possible cells?
- Success metric: Did we predict the correct cell?

### Model Pipeline
```
Raw AIS Data → H3 Sequences → 65 Features → Classifier → Next Cell Prediction
```

## Why Start Simple?

This focused approach gives us:
1. **Quick wins** - Clear success criteria
2. **Solid foundation** - Can extend to multi-step, fleet patterns later  
3. **Real value** - Vessel operators care about next destination
4. **Manageable scope** - Avoid complexity until we prove basic approach

## Dependencies

- **Core**: pandas, numpy, scikit-learn
- **Geospatial**: h3-py (hexagonal indexing)
- **Analysis**: jupyter, matplotlib, seaborn
- **Full list**: See `requirements.txt`

## Next Steps (Phase 3)

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
