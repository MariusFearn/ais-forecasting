# Basic Runtime Instructions

## Project Focus
**Simple Goal**: Predict which H3 cell a vessel will visit next using machine learning.

## Current Status
- ✅ Phase 1: H3 indexing and data foundation complete
- ✅ Phase 2: 65 vessel features extracted and validated  
- 🎯 Phase 3: Create simple ML classifier for next-cell prediction

## Quick Commands

### Environment Setup
```bash
cd /home/marius/repo_linux/ais-forecasting
pip install -r requirements.txt
```

### Data Processing
```python
# Load vessel features (Phase 2 output)
import pickle
with open('data/processed/vessel_features_sample.pkl', 'rb') as f:
    features = pickle.load(f)
```

### Next Steps (Phase 3)
1. Create training data: `scripts/create_training_data.py`
2. Train classifier: `scripts/train_model.py`  
3. Evaluate results: `scripts/evaluate_model.py`

## Success Criteria
- >60% accuracy predicting next H3 cell
- <15km average distance error
- Working visualization of predictions