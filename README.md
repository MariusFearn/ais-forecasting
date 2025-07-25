# AIS Forecasting Project

A comprehensive deep learning forecasting system for maritime AIS (Automatic Identification System) data that predicts vessel traffic patterns and maritime metrics using state-of-the-art time series models.

## Overview

This project implements a robust, modular forecasting pipeline that processes AIS data to generate accurate predictions of maritime traffic patterns. It features multiple deep learning architectures including Temporal Fusion Transformer (TFT) and N-BEATS models, with comprehensive feature engineering and evaluation capabilities.

## Features

- **Multiple Model Architectures**: Support for TFT, N-BEATS, and extensible framework for new models
- **Advanced Feature Engineering**: Geospatial (H3-based) and temporal feature creation with proper validation
- **Robust Data Pipeline**: Comprehensive data validation, cleaning, and preprocessing with missing data imputation
- **Hyperparameter Optimization**: Systematic optimization using Optuna with proper experiment tracking
- **Comprehensive Evaluation**: Multiple metrics, visualization tools, and proper time-series cross-validation
- **Production Ready**: Modular design with proper testing, documentation, and monitoring capabilities
- **Configuration Management**: Centralized configuration with inheritance for experiment reproducibility
- **Data Versioning**: Clear tracking of datasets used for each experiment

## Project Structure

```
ais-forecasting/
├── data/                       # Data storage
│   ├── raw/                    # Original AIS data
│   │   ├── ais_cape_data_2018.pkl  # Raw capsize vessel data (2018)
│   │   ├── ais_cape_data_2019.pkl  # Raw capsize vessel data (2019)
│   │   ├── ais_cape_data_2020.pkl  # Raw capsize vessel data (2020)
│   │   ├── ais_cape_data_2021.pkl  # Raw capsize vessel data (2021)
│   │   ├── ais_cape_data_2022.pkl  # Raw capsize vessel data (2022)
│   │   ├── ais_cape_data_2023.pkl  # Raw capsize vessel data (2023)
│   │   ├── ais_cape_data_2024.pkl  # Raw capsize vessel data (2024) ✅ Tested
│   │   └── ais_cape_data_2025.pkl  # Raw capsize vessel data (2025)
│   ├── processed/              # Preprocessed datasets
│   │   ├── README.md           # Documentation for processed data
│   │   └── vessel_features_sample.pkl  # ✅ Phase 2 output: 65 vessel features
│   └── models/                 # Serialized models
│       └── README.md           # Placeholder for model storage
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── evaluation.ipynb        # Placeholder - Model evaluation notebook
│   ├── exploratory.ipynb       # Placeholder - Initial data exploration
│   ├── model_development.ipynb # Placeholder - Model development
│   ├── preprocessing.ipynb     # Placeholder - Data preprocessing
│   └── vessel_exploration.ipynb # ✅ COMPLETE: Phase 1 & 2 vessel H3 analysis (37 cells)
├── src/                        # Source code
│   ├── __init__.py            # ✅ Package initialization with imports
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py        # ✅ Module exports (AISDataLoader, AISDataPreprocessor)
│   │   ├── investigate_data.py # 🔧 Partial: Data investigation utilities
│   │   ├── loader.py          # 🔧 Partial: AISDataLoader class structure
│   │   └── preprocessing.py   # 🔧 Partial: AISDataPreprocessor class structure
│   ├── features/               # Feature engineering ✅ COMPLETE
│   │   ├── __init__.py        # ✅ Module exports for all feature classes
│   │   ├── geo_features.py    # 🔧 Partial: GeoFeatureEngineer class structure
│   │   ├── time_features.py   # 🔧 Partial: TimeFeatureEngineer class structure
│   │   ├── vessel_features.py # ✅ COMPLETE: VesselFeatureExtractor (65 features)
│   │   └── vessel_h3_tracker.py # ✅ COMPLETE: VesselH3Tracker for H3 sequences
│   ├── models/                 # Model implementations
│   │   ├── __init__.py        # ✅ Module exports (BaseTimeSeriesModel, TFTModel, NBeatsModel)
│   │   ├── base_model.py      # 🔧 Partial: Abstract base class for models
│   │   ├── nbeats_model.py    # 🔧 Partial: N-BEATS model implementation
│   │   └── tft_model.py       # 🔧 Partial: Temporal Fusion Transformer model
│   ├── utils/                  # Utilities (metrics, optimization)
│   │   ├── __init__.py        # ✅ Module exports for metrics and optimization
│   │   ├── metrics.py         # 🔧 Partial: Evaluation metrics (MAE, RMSE, SMAPE, etc.)
│   │   └── optimize.py        # 🔧 Partial: Hyperparameter optimization with Optuna
│   └── visualization/          # Plotting and visualization
│       ├── __init__.py        # ✅ Module exports for plotting functions
│       └── plots.py           # 🔧 Partial: Forecasting visualization utilities
├── scripts/                    # Training and evaluation scripts
│   ├── __init__.py            # Empty initialization file
│   ├── evaluate.py            # 🔧 Partial: Model evaluation script structure
│   ├── predict.py             # 🔧 Partial: Prediction script structure
│   ├── quick_start_h3.py      # ✅ COMPLETE: Quick H3 exploration utility
│   └── train.py               # 🔧 Partial: Training script structure
├── tests/                      # Unit tests
│   ├── __init__.py            # ✅ Test suite documentation
│   ├── test_data.py           # 🔧 Partial: Tests for data processing
│   ├── test_features.py       # 🔧 Partial: Tests for feature engineering
│   └── test_models.py         # 🔧 Partial: Tests for model implementations
├── visualizations/            # Generated visualization outputs
│   ├── complete_global_maritime_heatmap.html    # Maritime traffic heatmap
│   ├── complete_maritime_3d_globe.html          # 3D globe visualization
│   ├── global_maritime_heatmap.html             # Global heatmap
│   ├── h3_hexagon_chess_board.html              # H3 hexagon visualization
│   ├── maritime_3d_globe.html                   # Maritime 3D globe
│   ├── maritime_regional_dashboard.html         # Regional dashboard
│   ├── maritime_traffic_animation.html          # Traffic animation
│   ├── ultra_fast_maritime_visualization.py     # 🔧 Partial: Fast processing script
│   └── vessel_h3_journey_exploration.html       # ✅ Vessel journey visualization
├── config/                     # Configuration files
│   ├── default.yaml           # Default configuration parameters
│   └── experiment_configs/     # Experiment-specific configurations
│       ├── nbeats_experiment.yaml  # N-BEATS model configuration
│       └── tft_experiment.yaml     # TFT model configuration
├── raw_data/                  # Symbolic link to data/raw/ (convenience)
├── requirements.txt           # ✅ COMPLETE: All dependencies including h3, geopy, holidays
├── README.md                  # ✅ This file - Project documentation
├── TOP_PLAN.md               # 📋 Project restructuring plan
├── todo_create_features.md   # ✅ COMPLETE: Phase 1-2 todo list (all checked)
├── PHASE_2_SUMMARY.md        # ✅ COMPLETE: Phase 2 accomplishments summary
├── create_features_h3.md     # Feature engineering documentation
└── data_summary.md           # Data analysis summary
```

**Legend:**
- ✅ **COMPLETE**: Fully implemented and tested
- 🔧 **Partial**: Structure exists, partial implementation
- 📋 **Documentation**: Planning and documentation files  
- No marker: Placeholder or empty files

**Current Status**: Phase 2 Complete - 65 vessel features successfully extracted and ready for sequence modeling (Phase 3).

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ais-forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your specific configuration
```

## Quick Start

### 1. Data Preparation

```python
from src.data.loader import AISDataLoader
from src.data.preprocessing import AISDataPreprocessor

# Load and preprocess data
loader = AISDataLoader('./data')
preprocessor = AISDataPreprocessor()

# Load raw data
df_raw = loader.load_raw_data('./data/raw/ais_data.csv')

# Clean and validate
df_clean = preprocessor.clean_ais_data(df_raw)

# Create features
df_features = preprocessor.create_h3_features(df_clean)
df_features = preprocessor.create_time_features(df_features)

# Save processed data
loader.save_processed_data(df_features, 'processed_ais_data')
```

### 2. Model Training

```bash
# Train a TFT model
python scripts/train.py --config config/experiment_configs/tft_experiment.yaml --model tft --data processed_ais_data

# Train an N-BEATS model
python scripts/train.py --config config/experiment_configs/nbeats_experiment.yaml --model nbeats --data processed_ais_data
```

### 3. Generate Predictions

```bash
python scripts/predict.py --config config/default.yaml --model-type tft --model-path ./data/models/tft_model.pth --data test_data --output predictions.pt
```

### 4. Evaluate Model

```bash
python scripts/evaluate.py --config config/default.yaml --model-type tft --model-path ./data/models/tft_model.pth --test-data test_data --output-dir ./results
```

## Configuration

The project uses YAML configuration files for managing parameters:

- `config/default.yaml`: Base configuration with default parameters
- `config/experiment_configs/`: Experiment-specific configurations

Key configuration sections:
- **model**: Model architecture parameters
- **training**: Training hyperparameters
- **data**: Data processing settings
- **features**: Feature engineering configuration
- **optimization**: Hyperparameter tuning settings

## Models

### Temporal Fusion Transformer (TFT)
- State-of-the-art attention-based architecture for multi-horizon forecasting
- Handles multiple input types (static, time-varying known/unknown features)
- Provides interpretability through attention mechanisms and variable importance
- Excellent for complex time series with multiple covariates
- Built-in feature selection and temporal pattern recognition

### N-BEATS
- Neural basis expansion analysis for interpretable time series forecasting
- Pure deep learning approach with minimal feature engineering requirements
- Excellent for univariate forecasting with trend and seasonality decomposition
- Hierarchical structure with forecast and backcast stacks
- Fast training and inference with strong empirical performance

### Future Model Support
The extensible framework is designed to support additional architectures:
- **DeepAR**: Probabilistic forecasting with autoregressive recurrent networks
- **Informer**: Efficient transformer for long sequence time-series forecasting
- **Prophet**: Facebook's forecasting tool for time series with strong seasonal patterns

### Extensible Framework
The `BaseTimeSeriesModel` abstract class allows easy integration of new models:

```python
from src.models.base_model import BaseTimeSeriesModel

class CustomModel(BaseTimeSeriesModel):
    def __init__(self, config):
        # Your implementation
        pass
    
    def fit(self, train_dataloader, val_dataloader):
        # Your training logic
        pass
    
    # Implement other required methods...
```

## Feature Engineering

### Geospatial Features (H3-based)
- H3 hexagonal grid cells for spatial discretization
- Distance and bearing calculations
- Speed estimation from position data
- Spatial aggregations and neighborhoods

### Temporal Features
- Cyclical encoding (hour, day, month)
- Lag and rolling window features
- Seasonal and calendar features
- Time-since-event features

## Data Management Pipeline

#### Preprocessing Features
- **H3 Geospatial Binning**: Consistent resolution handling with proper validation
- **Vessel Filtering**: Advanced data validation and quality checks
- **Missing Data Imputation**: Strategic approaches for handling incomplete data
- **Feature Normalization**: Standardization and scaling with proper statistics tracking
- **Data Versioning**: Clear tracking of datasets used for each experiment

#### Feature Engineering
- **Geospatial Features**: Advanced H3-based features with distance, bearing, and speed calculations
- **Time-Series Features**: Calendar features, rolling statistics, lag features, and seasonal components
- **Feature Selection**: Systematic approach to identify most predictive features with importance tracking

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Development

### Adding New Features
1. Create feature engineering functions in `src/features/`
2. Add corresponding tests in `tests/test_features.py`
3. Update configuration files if needed

### Adding New Models
1. Inherit from `BaseTimeSeriesModel`
2. Implement required abstract methods
3. Add model-specific configuration
4. Create tests in `tests/test_models.py`

### Code Quality
- Use black for code formatting: `black src/ tests/`
- Use flake8 for linting: `flake8 src/ tests/`
- Maintain test coverage above 80%

## Hyperparameter Optimization

The project includes Optuna integration for systematic hyperparameter tuning:

```python
from src.utils.optimize import HyperparameterOptimization

# Define parameter space
param_space = {
    'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
    'hidden_size': {'type': 'categorical', 'choices': [64, 128, 256]},
}

# Run optimization
optimizer = HyperparameterOptimization(
    trial_function=your_objective_function,
    param_space=param_space,
    n_trials=100
)
best_params = optimizer.optimize()
```

## Evaluation Metrics

The project supports multiple evaluation metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Symmetric Mean Absolute Percentage Error (SMAPE)
- Mean Absolute Percentage Error (MAPE)
- Quantile Loss

## Success Metrics and Performance

The project's effectiveness is measured through:

### Quantitative Metrics
- **Accuracy Improvement**: Target reduction in RMSE and MAE on validation data by at least 15%
- **Training Efficiency**: Decrease in training time by at least 20% through optimized pipelines
- **Deployment Consistency**: Reliable model performance across different environments
- **Development Speed**: Faster experimentation and model iteration cycles

### Qualitative Metrics
- **Code Quality**: Comprehensive test coverage (>80%) and documentation standards
- **Reproducibility**: Consistent results across different runs and environments
- **Maintainability**: Clear module interfaces and extensible architecture
- **Model Understanding**: Enhanced interpretability and feature importance insights

### Evaluation Framework
- **Time-Series Cross-Validation**: Proper temporal validation to prevent data leakage
- **Multiple Metrics**: MAE, RMSE, SMAPE, MAPE, and Quantile Loss for comprehensive assessment
- **Visualization Tools**: Interactive plots for forecast analysis and model interpretation
- **Backtesting**: Historical performance validation with sliding window approach

## Visualization

Comprehensive visualization tools for:
- Forecast vs actual comparisons
- Error distribution analysis
- Feature importance plots
- Attention weight heatmaps (for TFT)
- Metrics tracking over time

## Architecture Philosophy

This project follows a modular, maintainable structure based on best practices for ML projects. The architecture addresses common challenges in time series forecasting projects:

### Key Benefits

#### 1. Modularity and Reusability
- **Problem Solved**: Avoids monolithic code with hardcoded parameters
- **Solution**: Modularized components with clear interfaces allow for reuse and easier maintenance

#### 2. Configuration Management
- **Problem Solved**: Eliminates scattered parameters that make experiments hard to reproduce
- **Solution**: Centralized configuration with inheritance for comprehensive experiment tracking

#### 3. Data Pipeline Robustness
- **Problem Solved**: Prevents fragile preprocessing with inconsistent edge case handling
- **Solution**: Robust pipeline with proper validation, error handling, and data versioning

#### 4. Standardized Model Development
- **Problem Solved**: Eliminates ad-hoc model development with inconsistent training procedures
- **Solution**: Standardized model interface and training loop with proper validation and early stopping

#### 5. Comprehensive Evaluation Framework
- **Problem Solved**: Addresses limited evaluation metrics and visualization capabilities
- **Solution**: Comprehensive evaluation framework with proper time-series cross-validation and interpretability

## Advanced Capabilities

### Model Training and Optimization
- **Standardized Training Loop**: Consistent training with proper validation and checkpointing
- **Early Stopping**: Prevent overfitting with appropriate patience mechanisms
- **Model Checkpointing**: Save best models based on validation metrics
- **Experiment Tracking**: Comprehensive logging of metrics, parameters, and artifacts
- **Hyperparameter Optimization**: Systematic approach using Optuna with parallel trials

### Deployment and Monitoring
- **Model Serialization**: Standardized approach to save and load models
- **Prediction Pipeline**: Clean interface for generating forecasts in production
- **Performance Monitoring**: Track model performance over time with drift detection
- **Retraining Strategy**: Automated periodic model updates based on performance degradation

### Development Considerations
- **Data Drift Monitoring**: Systems to detect when model inputs deviate from training data
- **Online Learning**: Framework for incremental model updates
- **Ensemble Methods**: Support for combining different model architectures
- **Hardware Optimization**: Efficient utilization of available GPU/CPU resources
- **Explainability**: Enhanced model interpretability for business stakeholders

## Lessons Learned and Best Practices

This implementation incorporates lessons learned from previous maritime forecasting projects:

### Key Improvements
1. **Avoiding Overengineering**: Focus on proper abstraction rather than trying to do everything at once
2. **Data Quality First**: Comprehensive data validation and cleaning to ensure stable models
3. **Model Understanding**: Systematic approach to understanding model capabilities before implementation
4. **Systematic Optimization**: Structured hyperparameter selection using proven optimization techniques
5. **Proper Validation**: Time-series aware validation to get realistic performance estimates
6. **Feature Analysis**: Deep understanding of which features drive predictions for better model development

### Development Guidelines
- **Incremental Development**: Build and test components incrementally
- **Documentation First**: Document design decisions and trade-offs
- **Test-Driven Development**: Write tests for critical components before implementation
- **Configuration Management**: Use version-controlled configuration for all experiments
- **Monitoring Integration**: Build monitoring and alerting into the system from the start

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{ais-forecasting,
  title={AIS Forecasting: Deep Learning for Maritime Traffic Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/ais-forecasting}
}
```

## Contact

[Add your contact information here]
