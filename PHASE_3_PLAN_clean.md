# Phase 3: Improve ML Model Performance

## 🎯 Goal: Improve from 5% to >60% Accuracy

**Current Status**: Working Random Forest model with 5% test accuracy (91.8% train accuracy)  
**Target**: >60% next-cell prediction accuracy with <15km average error

## 📋 Immediate Next Steps

### 1. More Training Data (This Week)
- [ ] **Scale up vessels**: Process 5-10 vessels instead of 1
- [ ] **More sequences**: Generate 1000+ training examples instead of 199
- [ ] **Better features**: Use more of the 65 available features
- [ ] **Script**: Improve `scripts/create_simple_training_data.py`

### 2. Model Improvements (Next Week)  
- [ ] **Try XGBoost**: Often better than Random Forest for classification
- [ ] **Hyperparameter tuning**: Optimize tree depth, learning rate
- [ ] **Feature selection**: Identify most predictive of 65 features
- [ ] **Sequence features**: Use historical H3 cells, not just current

### 3. Evaluation & Visualization
- [ ] **Distance metrics**: Calculate km error, not just cell accuracy
- [ ] **Map visualization**: Show predicted vs actual vessel paths
- [ ] **Error analysis**: Where/when do predictions fail?
- [ ] **Script**: Create `scripts/evaluate_and_visualize.py`

## 🎯 Success Milestones

**Week 1**: 1000+ training examples, 20%+ accuracy  
**Week 2**: XGBoost model, 40%+ accuracy  
**Week 3**: Optimized model, 60%+ accuracy, visualization

## 🛠 Implementation Priority

1. **`scripts/create_training_data_scaled.py`** - Process multiple vessels
2. **`scripts/train_xgboost_model.py`** - Better model than Random Forest  
3. **`scripts/evaluate_model.py`** - Distance errors, visualizations

---

*Focus: Improve the working model rather than adding complexity.*
