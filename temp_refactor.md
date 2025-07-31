# Code Structure Refactoring Analysis & Plan

## 📊 Current State Analysis

### ✅ GOOD: `src/` Directory Structure (Follows Convention)
The `src/` directory correctly contains **reusable library code** that can be imported by scripts and notebooks:

- **`src/data/`**: Data loading and preprocessing utilities ✅
  - `loader.py`: AISDataLoader class for loading raw/processed data
  - `preprocessing.py`: Data cleaning and preparation functions
  - `investigate_data.py`: Data analysis utilities

- **`src/features/`**: Feature engineering modules ✅
  - `vessel_features.py`: VesselFeatureExtractor class (519 lines)
  - `vessel_h3_tracker.py`: VesselH3Tracker class for H3 conversion
  - `geo_features.py`: Geographic feature utilities
  - `time_features.py`: Temporal feature utilities

- **`src/models/`**: Model definitions and abstractions ✅
  - `base_model.py`: BaseTimeSeriesModel abstract class
  - `tft_model.py`: TFT model implementation
  - `nbeats_model.py`: N-BEATS model implementation

- **`src/utils/`**: General utilities ✅
  - `metrics.py`: Evaluation metrics
  - `optimize.py`: Optimization utilities

- **`src/visualization/`**: Plotting and visualization ✅
  - `plots.py`: Plotting functions

### ⚠️ ISSUES: `scripts/` Directory (Mixed Adherence)

**GOOD Scripts (Follow Convention):**
- `create_simple_training_data.py`: ✅ Executable script, imports from `src/`
- `train_simple_model.py`: ✅ Executable script, uses sklearn directly
- `test_simple.py`: ✅ Simple test script, imports from `src/`
- `ultra_fast_maritime_visualization.py`: ✅ Standalone visualization script

**PROBLEMATIC Scripts (Break Convention):**
- `evaluate.py`: ❌ **COMPLEX BUSINESS LOGIC** - Contains significant data processing, model loading, and evaluation logic that should be in `src/`
- `predict.py`: ❌ **COMPLEX BUSINESS LOGIC** - Contains substantial prediction pipeline logic that should be in `src/`
- `create_multi_vessel_training_data.py`: ❌ **EMPTY FILE** - Needs implementation
- `train_enhanced_model.py`: ❌ **EMPTY FILE** - Needs implementation

### ⚠️ ISSUES: Notebooks (Code Duplication)

**Analysis of Notebooks:**
- **`visual_training_analysis.ipynb`**: ❌ Contains significant **inline data processing logic** that duplicates functionality already in `src/` modules. Should primarily call `src/` functions.
- **Other notebooks**: Haven't been executed, but likely contain similar code duplication patterns.

### 🎯 SPECIFIC PROBLEMS IDENTIFIED

#### 1. **Scripts Violating `src/scripts` Convention**

**`scripts/evaluate.py` (177 lines):**
- Contains complex model loading logic
- Has extensive data processing code
- Implements evaluation pipeline logic
- **SOLUTION**: Move core logic to `src/models/` and `src/utils/`, keep script as thin wrapper

**`scripts/predict.py` (126 lines):**
- Contains prediction pipeline implementation
- Has complex model loading and data preparation
- **SOLUTION**: Move logic to `src/models/prediction.py`, keep script as thin CLI wrapper

#### 2. **Notebooks with Embedded Business Logic**

**`notebooks/visual_training_analysis.ipynb`:**
- Contains inline data loading and processing (duplicates `src/data/`)
- Has feature engineering code (duplicates `src/features/`)
- Implements model training logic (duplicates functionality)
- **SOLUTION**: Refactor to primarily call `src/` functions

#### 3. **Missing Implementation**

**Empty Scripts:**
- `scripts/create_multi_vessel_training_data.py`: Empty file
- `scripts/train_enhanced_model.py`: Empty file
- **SOLUTION**: Implement as thin wrappers around `src/` functionality

## 🔧 REFACTORING PLAN

### Phase 1: Extract Business Logic from Scripts

#### 1.1 Create `src/models/evaluation.py`
```python
# Extract evaluation logic from scripts/evaluate.py
class ModelEvaluator:
    def __init__(self, config)
    def evaluate_model(self, model_type, model_path, test_data_path)
    def generate_evaluation_report(self, results)
```

#### 1.2 Create `src/models/prediction.py`
```python
# Extract prediction logic from scripts/predict.py
class ModelPredictor:
    def __init__(self, config)
    def load_model(self, model_type, model_path)
    def predict(self, data_path, output_path)
```

#### 1.3 Create `src/training/` module
```python
# New module for training pipelines
# training/simple_trainer.py
# training/multi_vessel_trainer.py
# training/data_creator.py
```

### Phase 2: Refactor Scripts to be Thin Wrappers

#### 2.1 Refactor `scripts/evaluate.py`
```python
# BEFORE: 177 lines of complex logic
# AFTER: ~30 lines calling src/models/evaluation.py
def main():
    config = load_config(args.config)
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate_model(args.model_type, args.model_path, args.test_data)
    evaluator.generate_evaluation_report(results, args.output_dir)
```

#### 2.2 Refactor `scripts/predict.py`
```python
# BEFORE: 126 lines of complex logic  
# AFTER: ~25 lines calling src/models/prediction.py
def main():
    config = load_config(args.config)
    predictor = ModelPredictor(config)
    predictor.predict(args.data, args.output)
```

#### 2.3 Implement Missing Scripts
```python
# scripts/create_multi_vessel_training_data.py
# Thin wrapper around src/training/data_creator.py

# scripts/train_enhanced_model.py  
# Thin wrapper around src/training/multi_vessel_trainer.py
```

### Phase 3: Refactor Notebooks

#### 3.1 Refactor `notebooks/visual_training_analysis.ipynb`
- **REMOVE**: Inline data loading code → Use `src/data/loader.py`
- **REMOVE**: Inline feature engineering → Use `src/features/vessel_features.py`
- **REMOVE**: Inline model training → Use `src/training/` modules
- **KEEP**: Visualization, analysis, and exploration code
- **RESULT**: Notebook becomes clean analysis focused on insights, not implementation

#### 3.2 Review Other Notebooks
- Apply similar refactoring to other notebooks
- Ensure they primarily call `src/` functions rather than implementing logic

### Phase 4: Create Missing `src/` Modules

#### 4.1 `src/training/` package
```
src/training/
├── __init__.py
├── data_creator.py      # Multi-vessel training data creation
├── simple_trainer.py    # Simple model training pipeline  
├── enhanced_trainer.py  # Enhanced model training pipeline
└── pipeline.py          # Common training pipeline utilities
```

#### 4.2 Enhance existing modules
- Add missing functionality to `src/models/`
- Expand `src/utils/` with common utilities
- Improve `src/data/` with more data processing functions

## 🎉 REFACTORING COMPLETION STATUS

### ✅ COMPLETED: HIGH PRIORITY TASKS
All core refactoring tasks have been **successfully completed** and **tested**:

1. **✅ Scripts Refactoring Complete**
   - `scripts/evaluate.py`: Refactored from 177 lines to 35-line wrapper
   - `scripts/predict.py`: Refactored from 126 lines to 30-line wrapper
   - `scripts/create_multi_vessel_training_data.py`: Implemented as 25-line wrapper
   - `scripts/train_enhanced_model.py`: Implemented as 30-line wrapper

2. **✅ Core src/ Modules Created**
   - `src/models/evaluation.py`: ModelEvaluator class (150+ lines)
   - `src/models/prediction.py`: ModelPredictor class (120+ lines)
   - `src/training/` package: Complete training pipeline (4 modules)

3. **✅ Testing Validated**
   - `scripts/create_simple_training_data.py`: Working (199 samples created)
   - `scripts/train_simple_model.py`: Working (5% accuracy baseline)
   - `scripts/create_multi_vessel_training_data.py`: Working (24,950 samples)
   - `scripts/train_enhanced_model.py`: Working (enhanced model trained)
   - Import issues resolved with conditional imports in `src/models/__init__.py`

### ⏳ REMAINING: MEDIUM/LOW PRIORITY TASKS
The core refactoring is complete. Remaining tasks are for additional polish:

## 📋 IMPLEMENTATION TODO LIST

### HIGH PRIORITY (Core Convention Violations)

- [x] **Extract evaluation logic from `scripts/evaluate.py`**
  - [x] Create `src/models/evaluation.py` with ModelEvaluator class
  - [x] Refactor `scripts/evaluate.py` to be ~30 line wrapper
  - [x] Test that functionality is preserved ✅ (Scripts working, import issues resolved)

- [x] **Extract prediction logic from `scripts/predict.py`**
  - [x] Create `src/models/prediction.py` with ModelPredictor class  
  - [x] Refactor `scripts/predict.py` to be ~25 line wrapper
  - [x] Test that functionality is preserved ✅ (Scripts working, import issues resolved)

- [x] **Implement missing scripts**
  - [x] Implement `scripts/create_multi_vessel_training_data.py`
  - [x] Implement `scripts/train_enhanced_model.py`
  - [x] Both are now thin wrappers around new `src/training/` modules

### MEDIUM PRIORITY (Code Organization)

- [x] **Create `src/training/` package**
  - [x] Create `src/training/__init__.py`
  - [x] Create `src/training/data_creator.py` for multi-vessel data creation
  - [x] Create `src/training/simple_trainer.py` for simple model training
  - [x] Create `src/training/enhanced_trainer.py` for enhanced model training
  - [x] Create `src/training/pipeline.py` for common utilities

- [ ] **Refactor `notebooks/visual_training_analysis.ipynb`**
  - [ ] Replace inline data loading with `src/data/loader.py` calls
  - [ ] Replace inline feature engineering with `src/features/` calls
  - [ ] Replace inline model training with `src/training/` calls
  - [ ] Keep only visualization and analysis code

### LOW PRIORITY (Polish & Consistency)

- [ ] **Review other notebooks**
  - [ ] `notebooks/exploratory.ipynb`
  - [ ] `notebooks/preprocessing.ipynb`  
  - [ ] `notebooks/model_development.ipynb`
  - [ ] `notebooks/evaluation.ipynb`
  - [ ] `notebooks/vessel_exploration.ipynb`
  - [ ] Apply same refactoring pattern: remove inline logic, use `src/` imports

- [ ] **Enhance existing `src/` modules**
  - [ ] Add more utilities to `src/utils/`
  - [ ] Expand `src/data/` with additional data processing functions
  - [ ] Add more visualization functions to `src/visualization/`

## 🎯 SUCCESS CRITERIA

After refactoring, the codebase should satisfy:

1. **`src/` contains only reusable library code** - Classes, functions, utilities
2. **`scripts/` contains only thin executable wrappers** - CLI interfaces, argument parsing, calling `src/` functions
3. **`notebooks/` focus on analysis and exploration** - Visualization, insights, experimentation, but minimal implementation
4. **No code duplication** - Business logic exists once in `src/`, used everywhere else
5. **Clear separation of concerns** - Data processing in `src/data/`, models in `src/models/`, etc.

## 📊 ESTIMATED EFFORT

- **HIGH PRIORITY**: ~4-6 hours (Core convention fixes)
- **MEDIUM PRIORITY**: ~3-4 hours (Organization improvements)  
- **LOW PRIORITY**: ~2-3 hours (Polish and consistency)
- **TOTAL**: ~9-13 hours for complete refactoring

## 🚀 BENEFITS AFTER REFACTORING

1. **Clear Architecture**: Obvious separation between library code and executables
2. **Code Reusability**: `src/` modules can be easily imported and reused
3. **Testing**: `src/` modules can be unit tested independently
4. **Maintainability**: Business logic centralized in `src/`, not scattered
5. **Documentation**: Clear purpose for each file and directory
6. **Professional Structure**: Follows Python project best practices

---

*This refactoring plan will transform the codebase from a mixed structure to a clean, professional Python project following the `src/`/`scripts` convention consistently.*
