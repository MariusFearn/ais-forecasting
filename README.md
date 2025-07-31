

# AIS Vessel Trajectory Prediction

A simple machine learning system for predicting individual vessel movements using AIS data and H3 geospatial indexing.

## 🎯 Simple Goal
**Predict which H3 cell a vessel will visit next** based on its current position and movement patterns.

## ✅ Current Status

### COMPLETED (Phase 1-3):
- **Data**: 8 years Cape Town AIS data (2018-2025, 8.3M+ records)
- **H3 Indexing**: Resolution 5 (8.54km edge length) 
- **Vessel Tracking**: Convert GPS → H3 cell sequences
- **Feature Engineering**: ✅ **54 features implemented** - 42 high-quality features ready for training (training pipeline needs update)
- **Working ML Pipeline**: Random Forest classifier trained (91.8% train accuracy)
- **Code Refactoring**: ✅ Professional src/scripts architecture implemented
- **Multi-vessel Training**: ✅ Enhanced model with 24,950 training samples

### 🎯 CURRENT FOCUS (Phase 4):
- ✅ Simple model: 199 training sequences, 5% accuracy baseline
- ✅ Enhanced model: 24,950 sequences from 50 vessels, 0.9% accuracy
- 🎯 Improve model accuracy with better features/algorithms
- 🎯 Add comprehensive evaluation and visualization tools

## 🚀 Quick Start

### 1. Simple Model (Single Vessel) - Working ✅
This is the baseline model to verify the pipeline works end-to-end.
```bash
# Activate conda environment
conda activate ML

# Run the complete simple pipeline
python scripts/create_simple_training_data.py   # Create training data (199 samples)
python scripts/train_simple_model.py            # Train Random Forest model (5% accuracy)
```

### 2. Enhanced Model (Multi-Vessel) - Working ✅
This scales up to all available vessels for better prediction accuracy.
```bash
# Create comprehensive multi-vessel dataset  
python scripts/create_multi_vessel_training_data.py  # 24,950 samples from 50 vessels

# Train enhanced model with vessel-specific features
python scripts/train_enhanced_model.py               # Enhanced Random Forest (0.9% accuracy)
```

### 3. Evaluation & Prediction (Available)
```bash
# Evaluate trained models (requires lightning package)
python scripts/evaluate.py <model_path> --data-path <test_data>

# Make predictions with trained models (requires lightning package)  
python scripts/predict.py <model_path> --speed 10 --heading 90 --lat -33.9 --lon 18.4
```

### 2. Project Structure & Cleanup Plan
This is a detailed breakdown of the project structure with recommendations for cleanup.

```
ais-forecasting/
├── .github/                    # Contains GitHub-specific files, like CI/CD workflows.
│
├── config/                     # Stores all project configuration files.
│   ├── default.yaml            # Default parameters for the entire project.
│   └── experiment_configs/     # Configurations for specific machine learning experiments.
│       ├── nbeats_experiment.yaml # Settings for an N-BEATS model experiment.
│       └── tft_experiment.yaml    # Settings for a Temporal Fusion Transformer experiment.
│
├── data/                       # Holds all data used in the project.
│   ├── raw/                    # Raw, immutable data. Should not be modified.
│   ├── processed/              # Cleaned, transformed, and feature-engineered data.
│   │   ├── training_sets/      # Final datasets ready for model training.
│   │   ├── vessel_features/    # Intermediate features extracted for each vessel.
│   │   └── predictions/        # Stores the output predictions from models.
│   └── models/                 # Contains all trained model artifacts.
│       ├── final_models/       # Serialized, production-ready models.
│       ├── checkpoints/        # Saved states during large model training.
│       └── hyperparameter_logs/# Logs from hyperparameter optimization runs.
│
├── experiments/                # Tracks results and artifacts from ML experiments.
│   ├── baseline_experiments/   # Results from simple baseline models.
│   ├── nbeats_experiments/     # Results from N-BEATS model experiments.
│   └── tft_experiments/        # Results from TFT model experiments.
│
├── notebooks/                  # Jupyter notebooks for interactive analysis and visualization.
│   ├── exploratory.ipynb       # Initial data exploration and analysis.
│   ├── preprocessing.ipynb     # Interactive data cleaning and preparation.
│   ├── model_development.ipynb # Prototyping and developing new models.
│   ├── evaluation.ipynb        # In-depth evaluation of model performance.
│   └── visual_training_analysis.ipynb # Visualizing the full training pipeline.
│
├── scripts/                    # Contains standalone, executable scripts for core tasks.
│   ├── create_simple_training_data.py # Generates a small, single-vessel dataset.
│   ├── train_simple_model.py   # Trains a baseline model on the simple dataset.
│   ├── create_multi_vessel_training_data.py # Generates the full training dataset.
│   ├── train_enhanced_model.py # Trains the primary, enhanced model.
│   ├── evaluate.py             # Runs model evaluation from the command line.
│   ├── predict.py              # Runs predictions using a trained model.
│   └── test_simple.py          # A simple test script for quick validation.
│
├── src/                        # Contains all the project's source code as a Python package.
│   ├── __init__.py             # Makes 'src' a package, allowing imports.
│   ├── data/                   # Modules for data loading and preprocessing.
│   ├── features/               # Modules for feature engineering and transformation.
│   ├── models/                 # Python definitions of model architectures.
│   ├── utils/                  # Reusable utility functions and helper classes.
│   └── visualization/          # Code for generating plots and maps.
│
├── tests/                      # Contains all tests for the project source code.
│   ├── test_data.py            # Unit tests for data loading and validation.
│   ├── test_features.py        # Unit tests for the feature engineering pipeline.
│   └── test_models.py          # Unit tests for model input/output validation.
│
├── visualizations/             # Stores saved output plots, maps, and other visuals.
│   ├── *.html                  # Interactive maps and plots generated by notebooks/scripts.
│   └── ultra_fast_maritime_visualization.py # Move: This is a script, not a visualization.
│
├── README.md                   # This file: The main documentation for the project.
├── requirements.txt            # A list of all Python packages required to run the project.
└── .gitignore                  # Specifies files and folders to be ignored by Git.
```


## 📊 Data Summary

### AIS Data Files (data/raw/):
- **8 files**: `ais_cape_data_2018.pkl` to `ais_cape_data_2025.pkl`
- **Format**: Pandas DataFrames with 18 columns
- **Sample size**: 8.3M+ records (2018), ~1.1GB per year
- **Coverage**: Cape Town maritime area, UTC timestamps
- **Key columns**: `imo` (vessel ID), `lat/lon` (position), `speed`, `heading`, `mdt` (timestamp)

### Feature Engineering (54 Features - Implemented & Working)
**✅ IMPLEMENTED**: The feature engineering pipeline now extracts **54 real features** with **42 high-quality features** ready for training.

**Currently Implemented Categories:**

**Basic State Features (6 features):**
- `current_h3_cell`, `current_speed`, `current_heading`, `lat`, `lon`, `time_in_current_cell`

**Historical Sequence Features (14 features):**
- `cells_visited_6h/12h/24h`, `avg_speed_6h/12h/24h`, `cell_transitions_6h`
- `time_in_cell_hours`, `cells_visited_cumulative`, `cell_group`

**Movement Pattern Features (9 features):**
- `speed_trend_6h/12h`, `speed_std_6h/12h`, `heading_consistency_6h/12h`
- `delta_distance`, `est_speed`, `cell_transition`

**Journey Characteristics (6 features):**
- `total_journey_time`, `distance_from_start_km`, `journey_phase`
- Port detection: `likely_port_departure`, `likely_port_approach`

**Geographic Features (1 feature):**
- `ocean_region` (Cape Town area classification)

**Operational Features (7 features):**
- `hour_of_day`, `day_of_week`, `is_weekend`
- AIS metadata: `nav_status`, `destination`, `eta`

**Vessel Metadata (11 features):**
- Vessel identification, position history, timestamps, draught

### 🔮 Suggested Future Features (Advanced Implementation)
These features would require external data sources or complex domain knowledge:

**Enhanced Geographic Context:**
- Coastal proximity (requires coastline data)
- Water depth estimation (requires bathymetry data)  
- Shipping lane detection (requires traffic pattern data)
- Port vicinity indicators (requires port database)

**Advanced Operational Context:**
- Cargo status estimation (requires vessel type analysis)
- Weather impact features (requires weather API)
- Fuel efficiency patterns (requires engine data)
- Tide and current effects (requires oceanographic data)

**Sophisticated Movement Analysis:**
- Multi-step prediction sequences
- Fleet behavior patterns
- Route optimization metrics
- Anomaly detection features

## 🤖 Machine Learning Pipeline

### Current Working Example:
```
Raw AIS Data → H3 Sequences → 54 Features (42 high-quality) → Random Forest → Next Cell Prediction
```
**Challenge**: Training pipeline currently only uses 6-9 basic features instead of all 42 available high-quality features.

### Model Performance:
- **Simple Model (1 vessel)**: 91.8% train, 5.0% test accuracy (162 cells)
- **Enhanced Model (50 vessels)**: 55.2% train, 0.9% test accuracy (3,642 cells)
- **Challenge**: Large number of possible H3 cells makes prediction difficult
- **Feature Importance**: Location (lat/lon) most important, then current H3 cell

### Success Criteria:
- **Target**: >60% accuracy predicting next H3 cell
- **Distance**: <15km average error from actual position
- **Current Gap**: Need better features or different modeling approach

## 🔧 Technical Implementation

### Next Steps (Phase 4):
1. **Update training pipeline** - Use all 42 high-quality features instead of just 6-9 basic ones
2. **Feature selection** - Determine which of the 42 features are most predictive
3. **Better algorithms** - Try XGBoost, ensemble methods with complete feature set
4. **Evaluation framework** - Comprehensive metrics and distance-based evaluation
5. **Advanced features** - Implement suggested future features requiring external data

## 📦 Dependencies

**Core**: pandas, numpy, scikit-learn, h3-py  
**Geospatial**: geopandas, folium  
**ML**: pytorch, optuna (for advanced models)  
**Analysis**: jupyter, matplotlib, seaborn

Install: `pip install -r requirements.txt`

## 🎯 Why This Approach Works

1. **Clear Success Metrics** - Easy to measure if predictions are correct
2. **Incomplete Foundation** - Feature engineering framework exists but only basic features implemented
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

## 🔍 Code Analysis & Current Status

### ✅ **FEATURE ENGINEERING STATUS**: Complete & Working

**ACTUAL IMPLEMENTATION:** **54 features** with **42 high-quality features** ready for training

#### **What's Actually Implemented:**

**Complete Feature Set (54 features):**
- **High Quality (42 features)**: Real calculated values with good variance
- **Limited Use (12 features)**: Constant values or binary flags (vessel ID, status flags)

**Feature Categories Working:**
- ✅ **Basic State**: 6 features (position, speed, heading, time)
- ✅ **Historical Sequences**: 14 features (rolling windows, cumulative metrics)  
- ✅ **Movement Patterns**: 9 features (trends, variability, transitions)
- ✅ **Journey Characteristics**: 6 features (time, distance, phases)
- ✅ **Geographic Context**: 1 feature (regional classification)
- ✅ **Operational Context**: 7 features (time-based, AIS metadata)

#### **Current Bottleneck: Training Pipeline**
The **feature extraction works perfectly** but the **training pipeline ignores most features**:

**Training Uses:** Only 6-9 basic features  
**Available:** 42 high-quality features ready for use  
**Gap:** Training pipeline needs update to use all features

#### **Performance Issues Explained:**
- **Simple Model**: 5% accuracy - using only 6 features out of 42 available
- **Enhanced Model**: 0.9% accuracy - using only 9 features out of 42 available  
- **Root Cause**: Training pipeline limitation, not feature engineering

#### **Immediate Priority:**
1. ✅ **Feature engineering**: Complete and working (54 features)
2. 🎯 **Training pipeline**: Update to use all 42 high-quality features
3. 🎯 **Feature selection**: Identify most predictive features
4. 🎯 **Model optimization**: Better algorithms with complete feature set

### 🎯 **Updated Current Status:**
- ✅ **Data pipeline**: Working H3 conversion and comprehensive feature extraction
- ✅ **Code architecture**: Clean src/scripts structure implemented  
- ✅ **Feature engineering**: Complete - 54 features with 42 high-quality
- 🎯 **Training optimization**: Use all available features instead of basic subset
- 🎯 **Model improvement**: Expected significant accuracy gains with full feature set

---

*Focus: Utilize the comprehensive feature set that's already implemented.*
