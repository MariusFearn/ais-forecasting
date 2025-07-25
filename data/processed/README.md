# Processed Data Directory

This directory contains cleaned and feature-engineered datasets ready for model training.

## Data Processing Pipeline

1. **Raw data cleaning** (validation, outlier removal)
2. **Feature engineering** (H3 cells, temporal features)
3. **Data aggregation** (grid-based aggregation)
4. **Normalization** (feature scaling)

## File Naming Convention

- `processed_ais_data_YYYY-MM.pkl`: Monthly processed datasets
- `training_data_vXX.pkl`: Training-ready datasets
- `validation_data_vXX.pkl`: Validation datasets
- `test_data_vXX.pkl`: Test datasets

## Data Format

Processed data is stored as pickled pandas DataFrames with standardized columns:

### Core Columns
- `time_idx`: Sequential time index for modeling
- `vessel_id`: Vessel identifier
- `h3_cell`: H3 geospatial cell identifier
- `target_value`: Target variable for forecasting

### Feature Columns
- Temporal features (`hour`, `day_of_week`, `month`, etc.)
- Geospatial features (`h3_center_lat`, `h3_center_lon`, etc.)
- Derived features (`speed`, `distance_features`, etc.)

## Quality Metrics

Each processed dataset includes:
- Data quality report
- Feature distribution statistics
- Missing value analysis
