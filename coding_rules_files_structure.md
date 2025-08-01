# Coding Rules and Project Structure

## 🎯 Coding Standards and Rules

### **File Path Handling**
- **ALWAYS** use `pathlib.Path` for all file and directory paths
- Ensures cross-platform compatibility (Linux/Windows/Mac)
- Example: `from pathlib import Path; data_path = Path("data/raw")`

### **Logging Standards**
- **ALWAYS** use `logging` module for output in scripts and source code
- **NEVER** use `print()` for logging purposes
- Configure logging levels appropriately (DEBUG, INFO, WARNING, ERROR)

### **Code Formatting**
- **Adhere to PEP 8 standards** for all Python code
- Use type hints for all function signatures
- Include comprehensive docstrings with Args, Returns, and Raises sections

### **Environment Management**
- **ALWAYS** use `conda activate ML` environment (not base)
- Pin dependencies to exact versions in `requirements.txt`
- Use `pip-tools` for dependency management

---

## 📁 Project Structure Reference

```
ais-forecasting/
├── .github/                    # GitHub-specific files (CI/CD workflows)
├── config/                     # All project configuration files
│   ├── default.yaml            # Base parameters for entire project
│   ├── dl_default.yaml         # Deep learning specific defaults
│   └── experiment_configs/     # Experiment-specific configurations
├── data/                       # All project data
│   ├── raw/                    # Immutable original data (.pkl files)
│   │   └── parquet/            # Optimized parquet versions
│   ├── processed/              # Cleaned, transformed data
│   │   ├── training_sets/      # ML-ready datasets
│   │   ├── vessel_features/    # Individual vessel features
│   │   ├── fleet_features/     # Fleet-level aggregated features
│   │   └── predictions/        # Model output predictions
│   └── models/                 # Trained model artifacts
│       ├── final_models/       # Production-ready models
│       ├── checkpoints/        # Training state saves
│       └── hyperparameter_logs/# Optimization experiment logs
├── experiments/                # ML experiment tracking and results
├── notebooks/                  # Interactive analysis and research
├── scripts/                    # Executable scripts for core tasks
├── src/                        # Source code package (all functions here)
├── tests/                      # Unit and integration tests
└── visualizations/             # Generated plots, maps, and visuals
```

---

## 🔧 Source Code (`src/`) Organization

### **Data Processing (`src/data/`)**
- **`loader.py`**: Multi-year AIS data loading with DuckDB optimization
- **`duckdb_engine.py`**: High-performance SQL engine (10-50x speedup)
- **Core Functions**:
  - `AISDataLoader.load_multi_year_data()`: Load multiple years efficiently
  - `DuckDBEngine.execute_query()`: Optimized SQL operations

### **Feature Engineering (`src/features/`)**
- **`trajectory_processor.py`**: H3-based trajectory processing
- **`vessel_features.py`**: Individual vessel behavior analysis
- **Core Functions**:
  - `extract_vessel_trajectories()`: Convert AIS to trajectory segments
  - `process_trajectories_batch()`: Batch processing for memory efficiency

### **Machine Learning (`src/models/`)**
- **`clustering.py`**: DTW distance computation and route clustering
- **Core Functions**:
  - `compute_dtw_distance_matrix()`: Trajectory similarity analysis
  - `cluster_routes()`: DBSCAN clustering for route discovery

### **Training Pipeline (`src/training/`)**
- **Unified ML pipeline components**
- **GPU-accelerated XGBoost training**

### **Utilities (`src/utils/`)**
- **`logging_setup.py`**: Consistent logging configuration
- **Core helper functions and shared utilities**

### **Visualization (`src/visualization/`)**
- **Interactive map generation**
- **Maritime traffic visualization**

---

## 🎮 Backend Systems

### **DuckDB Engine** (Primary - Ultra Fast)
- **Purpose**: 10-50x speedup for data aggregations
- **Usage**: `from src.data.duckdb_engine import DuckDBEngine`
- **Benefits**: Memory efficient, SQL-based operations
- **Use Cases**: Large dataset processing, complex aggregations

### **AISDataLoader** (Existing Infrastructure)
- **Purpose**: Optimized multi-year AIS data loading
- **Usage**: `from src.data.loader import AISDataLoader`
- **Features**: Parquet support, memory-mapped arrays
- **Performance**: Handles 14.5M+ records efficiently

### **H3 Geospatial System**
- **Purpose**: Spatial indexing and trajectory analysis
- **Resolution**: Level 5 (8.54km edge length)
- **Usage**: Built into trajectory processing functions

---

## ⚙️ Configuration System

### **Base Configuration (`config/default.yaml`)**
```yaml
data:
  raw_data_path: "data/raw"
  processed_path: "data/processed"

processing:
  h3_resolution: 5
  memory_limit_gb: 45

model:
  type: "xgboost"
  device: "cuda:0"  # GPU acceleration
```

### **Experiment Configurations (`config/experiment_configs/`)**
- **`comprehensive_h3_experiment.yaml`**: Full-scale H3 analysis
- **`maritime_discovery.yaml`**: Production maritime discovery
- **`nbeats_experiment.yaml`**: N-BEATS model experiments
- **`tft_experiment.yaml`**: Temporal Fusion Transformer

### **Configuration Inheritance**
- Experiment configs inherit from `default.yaml`
- Override specific parameters for each experiment
- Reduces duplication by 55-63%

---

## 📜 Script Functions (`scripts/`)

### **Data Creation Scripts**
- **`create_training_data.py`**: Generate ML-ready datasets
- **`convert_to_parquet.py`**: Convert pickle to optimized parquet
- **`benchmark_duckdb.py`**: Performance comparison testing

### **Training Scripts**
- **`train_enhanced_model.py`**: Main ML model training
- **`train_h3_model.py`**: H3-specific model training

### **Maritime Discovery Scripts**
- **`maritime_discovery.py`**: Production maritime analysis pipeline
- **`test_maritime_discovery.py`**: Fast integration testing

### **Evaluation Scripts**
- **`evaluate_model.py`**: Model performance evaluation
- **`predict.py`**: Generate predictions using trained models
- **`phase4_summary.py`**: Comprehensive evaluation reporting

### **System Scripts**
- **`test_system.py`**: System validation and health checks
- **`setup_optimized_environment.sh`**: Environment configuration

---

## 🗄️ Raw Data Structure (`data/raw/`)

### **AIS Data Files**
```
data/raw/
├── ais_cape_data_2018.pkl    # 1.6 GB - Baseline year
├── ais_cape_data_2019.pkl    # 1.7 GB - +6% increase
├── ais_cape_data_2020.pkl    # 1.9 GB - +12% increase
├── ais_cape_data_2021.pkl    # 2.0 GB - +5% increase
├── ais_cape_data_2022.pkl    # 2.2 GB - +10% increase
├── ais_cape_data_2023.pkl    # 2.5 GB - +14% increase
├── ais_cape_data_2024.pkl    # 2.6 GB - +4% increase
└── parquet/                  # Optimized versions (3-5x faster loading)
    ├── ais_cape_data_2018.parquet
    ├── ais_cape_data_2019.parquet
    └── ...
```

### **Data Specifications**
- **Total Size**: ~16GB raw data
- **Record Count**: 14.5M+ AIS records
- **Time Span**: 2018-2024 (7+ years)
- **Vessel Coverage**: 1,794+ unique vessels
- **Column Schema**: `imo`, `mdt`, `lat`, `lon`, `speed`, `heading`

---

## 🧪 Testing Structure (`tests/`)

### **Test Organization**
- **`test_data.py`**: Data loading and validation tests
- **`test_features.py`**: Feature engineering pipeline tests
- **`test_models.py`**: Model input/output validation tests
- **`test_integration.py`**: End-to-end pipeline tests
- **`test_duckdb_integration.py`**: DuckDB optimization tests

### **Testing Standards**
- Use `unittest` framework for all tests
- Mock external dependencies where appropriate
- Include performance regression tests
- Validate output data quality and format

---

## 🔄 Development Workflow

### **The Golden Path**
1. **Add Core Logic**: Place functions in appropriate `src/` module
2. **Create/Update Config**: Add parameters to YAML files
3. **Update Scripts**: Modify scripts to use new logic and configs
4. **Add Tests**: Create corresponding tests in `tests/`
5. **Update Documentation**: Document in README and relevant files

### **Code Quality Checklist**
- [ ] Use `pathlib.Path` for all file operations
- [ ] Add logging instead of print statements
- [ ] Include type hints and docstrings
- [ ] Follow PEP 8 formatting standards
- [ ] Pin exact dependency versions
- [ ] Add unit tests for new functionality
- [ ] Update relevant configuration files
