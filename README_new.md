# AIS Vessel Trajectory Prediction

A professional machine learning system for predicting vessel movements using AIS data and H3 geospatial indexing.

## 🎯 Goal
**Predict which H3 cell a vessel will visit next** based on current position and movement patterns.

## ⚡ Quick Start
```bash
# 1. Setup environment
conda activate ML
pip install -r requirements.txt

# 2. Run system test (< 10 seconds)
python scripts/test_maritime_discovery.py

# 3. Try the complete ML pipeline
jupyter notebook notebooks/intro_to_ml.ipynb
```

## 🏆 Current Achievements
- **🎯 85.5% Prediction Accuracy** - 17x improvement from baseline
- **⚡ Maritime Discovery Pipeline** - 44 trajectories + 218 terminals in 9.1 seconds
- **🌍 Global Terminal Discovery** - 697 maritime terminals identified worldwide
- **🚀 GPU Acceleration** - RTX 3080 Ti optimized training pipeline
- **📊 Professional Architecture** - Configuration-driven, zero-duplication system

## 📊 System Capabilities

### **Data Processing**
- **8 years AIS data** (2018-2025, 14.5M+ records)
- **DuckDB optimization** (14x speedup over pandas)
- **Multi-vessel tracking** (1,794+ unique vessels)
- **H3 geospatial indexing** (Resolution 5, 8.54km cells)

### **Machine Learning**
- **XGBoost pipeline** with 54 engineered features
- **GPU acceleration** (RTX 3080 Ti support)
- **Configuration-driven** experiments (YAML-based)
- **Professional MLOps** structure

### **Maritime Discovery**
- **Real-time trajectory extraction** (H3-based spatial indexing)
- **Terminal identification** (behavioral analysis)
- **Route clustering** (DTW similarity analysis)
- **Global coverage** (all major oceans and shipping routes)

## 🚀 Core Pipelines

### **1. ML Training Pipeline**
```bash
# Train prediction model
python scripts/train_enhanced_model.py \
    --config config/experiment_configs/comprehensive_h3_experiment.yaml
```

### **2. Maritime Discovery Pipeline**
```bash
# Discover terminals and routes
python scripts/maritime_discovery.py \
    --config config/maritime_discovery.yaml \
    --output-dir ./data/processed/maritime_discovery
```

### **3. Model Evaluation**
```bash
# Evaluate model performance
python scripts/evaluate_model.py \
    --model-path data/models/final_models/xgboost_model.pkl \
    --test-data data/processed/training_sets/test_features.parquet
```

## 📁 Project Structure
```
ais-forecasting/
├── config/                 # YAML configuration files
├── data/                   # Raw and processed data
├── scripts/                # Executable pipeline scripts
├── src/                    # Source code modules
├── tests/                  # Unit and integration tests
├── notebooks/              # Interactive analysis
└── visualizations/         # Generated maps and plots
```

## 🔧 Technical Stack
- **Python 3.9+** with conda environment management
- **DuckDB** for high-performance data processing
- **XGBoost** with CUDA GPU acceleration
- **H3** for geospatial indexing and trajectory analysis
- **Folium** for interactive maritime visualization

## 📚 Documentation
- **[Installation & Setup](installation_setup_run_test.md)** - Quick start guide
- **[Coding Rules & Structure](coding_rules_files_structure.md)** - Development guidelines
- **[Project History](project_history_and_steps.md)** - Development timeline and achievements
- **[Maritime Discovery](MARITIME_DISCOVERY.md)** - Production pipeline details

## 🧪 Testing
```bash
# Quick system validation
python scripts/test_system.py

# Integration testing
python scripts/test_maritime_discovery.py

# Run unit tests
python -m pytest tests/
```

## 📈 Performance
- **Processing Speed**: 9.1 seconds for 5 vessels (40k records)
- **Prediction Accuracy**: 85.5% (17x improvement from baseline)
- **Global Discovery**: 697 terminals in 1.2 minutes
- **Hardware Optimization**: GPU + 14-thread CPU utilization

## 🎮 Interactive Demo
```bash
# Experience the complete system in action
jupyter notebook notebooks/intro_to_ml.ipynb
```
*Beginner-friendly notebook with visualizations and real-time pipeline execution*

## 🔗 Quick Links
- [Quick Start](installation_setup_run_test.md#quick-testing)
- [Configuration Guide](coding_rules_files_structure.md#configuration-system)
- [Performance Benchmarks](installation_setup_run_test.md#performance-benchmarks)
- [Troubleshooting](installation_setup_run_test.md#troubleshooting)
