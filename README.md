

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

### 1. Simple Model (Single Vessel)
This is the baseline model to verify the pipeline.
```bash
# Run our current working example
python scripts/create_simple_training_data.py   # Create training data (1 vessel)
python scripts/train_simple_model.py            # Train ML model
```

### 2. Enhanced Model (All Vessels)
This scales up the training to all available vessels for a better model.
```bash
# Create the comprehensive dataset
python scripts/create_multi_vessel_training_data.py

# Train the enhanced model
python scripts/train_enhanced_model.py
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
│   ├── test_simple.py          # A simple test script for quick validation.
│   ├── create_training_data.py # Delete: Redundant, functionality is split.
│   ├── train.py                # Delete: Redundant, functionality is split.
│   └── quick_start_h3.py       # Delete: Old script, functionality now in notebooks.
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

**Cleanup Plan:**
1.  **Delete redundant root files**: `PHASE_2_SUMMARY.md`, `PHASE_3_PLAN.md`, `PHASE_3_PLAN_clean.md`, `PHASE_3_SIMPLE.md`.
2.  **Delete redundant scripts**: `scripts/create_training_data.py`, `scripts/train.py`, `scripts/quick_start_h3.py`.
3.  **Delete redundant `__init__` files**: `src/models/__init___new.py`, `src/utils/__init___new.py`.
4.  **Move misplaced script**: Move `visualizations/ultra_fast_maritime_visualization.py` to `scripts/`.
5.  **Modify `.gitignore` file** to exclude `__pycache__` and other temporary files from version control.
```

## 📊 Data Summary

### AIS Data Files (data/raw/):
- **8 files**: `ais_cape_data_2018.pkl` to `ais_cape_data_2025.pkl`
- **Format**: Pandas DataFrames with 18 columns
- **Sample size**: 8.3M+ records (2018), ~1.1GB per year
- **Coverage**: Cape Town maritime area, UTC timestamps
- **Key columns**: `imo` (vessel ID), `lat/lon` (position), `speed`, `heading`, `mdt` (timestamp)

### Feature Engineering (65 features)
The feature engineering pipeline produces 65 detailed features for each vessel time step, categorized as follows:

- **Core State Features (9 features)**: `current_h3_cell`, `current_speed`, `current_heading`, `time_in_current_cell`, and other state indicators.
- **Historical Sequence Features (8 features)**: `cells_visited_6h/12h/24h`, `avg_speed_6h/12h/24h`, and rate of cell changes.
- **Movement Pattern Features (12 features)**: `speed_trend_6h/12h`, `speed_std_6h/12h`, `heading_consistency_6h/12h`, and movement efficiency.
- **Journey Characteristics (3 features)**: `total_journey_time`, `distance_from_start_km`, and cumulative cells visited.
- **Contextual Features (6 features)**: `journey_phase` (e.g., 'transit'), `likely_cargo_status`, and coastal proximity.
- **Advanced Features (27 features)**: Lag features at 1h, 6h, 12h intervals and other rolling statistics.

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
