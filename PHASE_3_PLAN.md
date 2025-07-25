# Phase 3: Simple ML Training Pipeline

## 🎯 **FOCUSED GOAL: Get First Working ML Model**

Based on completed Phase 2 (65 vessel features), create a simple classifier that predicts which H3 cell a vessel will visit next.

## 📋 **Phase 3 Tasks - Simple & Achievable**

### Week 3: Create Training Data
- [ ] **Convert features to ML datasets**
  - [ ] Input: Current vessel state + 65 features
  - [ ] Target: Next H3 cell the vessel actually visited
  - [ ] Create train/validation/test splits by time

- [ ] **Data preparation script**
  - [ ] `scripts/create_training_data.py`
  - [ ] Process multiple vessels from 2024 data
  - [ ] Save as `data/processed/training_sets/vessel_next_cell.pkl`

### Week 4: Train Simple Model
- [ ] **Simple classifier implementation**
  - [ ] Start with Random Forest (easy to interpret)
  - [ ] Input: 65 features → Output: H3 cell class
  - [ ] `src/models/vessel_classifier.py`

- [ ] **Training script**
  - [ ] `scripts/train_model.py`
  - [ ] Basic hyperparameter tuning
  - [ ] Save trained model to `data/models/`

- [ ] **Evaluation script**
  - [ ] `scripts/evaluate_model.py`
  - [ ] Classification accuracy
  - [ ] Distance error (km from actual)
  - [ ] Visualize predictions on map

## 🎯 **Success Criteria (Simple & Clear)**

1. **Technical Success**: >60% accuracy predicting next H3 cell
2. **Distance Success**: Average error <15km from actual position  
3. **Visual Success**: Can show predicted vs actual vessel paths
4. **Code Success**: Clean, runnable scripts for train/evaluate

## 🛠 **Implementation Plan**

### Phase 3.1: Data Preparation (3 days)
```python
# Create training sequences
def create_training_sequences(vessel_features):
    """Convert vessel features to input-target pairs"""
    sequences = []
    for t in range(len(vessel_features)-1):
        input_features = vessel_features[t]    # Current state (65 features)
        target_cell = vessel_features[t+1]['current_h3_cell']  # Next cell
        sequences.append((input_features, target_cell))
    return sequences
```

### Phase 3.2: Simple Model (2 days)
```python
# Simple classifier
from sklearn.ensemble import RandomForestClassifier

class VesselNextCellPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
    
    def fit(self, X_features, y_next_cells):
        self.model.fit(X_features, y_next_cells)
    
    def predict(self, X_features):
        return self.model.predict(X_features)
```

### Phase 3.3: Evaluation (2 days)
```python
# Simple evaluation
def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate distance errors
    distances = []
    for true_cell, pred_cell in zip(y_true, y_pred):
        true_lat, true_lon = h3.h3_to_geo(true_cell)
        pred_lat, pred_lon = h3.h3_to_geo(pred_cell)
        dist_km = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
        distances.append(dist_km)
    
    return {
        'accuracy': accuracy,
        'mean_distance_error_km': np.mean(distances),
        'median_distance_error_km': np.median(distances)
    }
```

## 📁 **File Structure for Phase 3**

```
scripts/
├── create_training_data.py     # Convert features to ML datasets
├── train_model.py              # Train simple classifier  
└── evaluate_model.py           # Evaluate and visualize results

src/models/
├── vessel_classifier.py        # Simple RandomForest classifier
└── base_model.py              # Simple base class

data/processed/training_sets/
├── vessel_next_cell_train.pkl  # Training data
├── vessel_next_cell_val.pkl    # Validation data
└── vessel_next_cell_test.pkl   # Test data

data/models/
├── vessel_classifier_v1.pkl    # Trained model
└── evaluation_results.json     # Performance metrics
```

## 🚀 **Why This Will Work**

1. **Builds on Phase 2**: Uses existing 65 features (proven to work)
2. **Simple problem**: Classification is easier than regression
3. **Clear success**: Easy to measure if it's working
4. **Fast iteration**: Can try different models quickly
5. **Visual validation**: Can see if predictions make sense on map

## 🎯 **After Phase 3 Success**

Once we have a working simple model, we can extend to:
- Multi-step prediction (next 3-5 cells)
- Different model types (XGBoost, Neural Networks)
- Fleet-level patterns
- Real-time prediction API

**But first: Get one simple thing working really well!**

---

*Focus: Simple vessel next-cell prediction that actually works.*
