# AIS Data Summary

## Overview
This document summarizes the structure and characteristics of the AIS (Automatic Identification System) data used for ship forecasting. The data consists of pickle files containing DataFrame objects with ship movement and trajectory information around Cape Town area.

## Data Files
Located in `raw_data/` directory:
- `ais_cape_data_2018.pkl`
- `ais_cape_data_2019.pkl` 
- `ais_cape_data_2020.pkl`
- `ais_cape_data_2025.pkl`

## Data Structure

### Format
- **File Type**: Pickle files containing pandas DataFrames
- **Sample Size**: 8,345,365 records (2018 data)
- **Columns**: 18 features per record
- **Memory Usage**: ~1.1 GB estimated for 2018 data

### Schema Description

| Column | Data Type | Description |
|--------|-----------|-------------|
| `imo` | int64 | International Maritime Organization number (unique ship identifier) |
| `lat1` | float64 | Previous latitude position |
| `lon1` | float64 | Previous longitude position |
| `draught` | float64 | Ship's draught (depth of ship below waterline) in meters |
| `speed` | float64 | Ship's speed over ground in knots |
| `mdt` | datetime64[ns, UTC] | Message DateTime - current timestamp |
| `next_mdt` | datetime64[ns, UTC] | Next message datetime (for trajectory prediction) |
| `destination` | object | Processed destination port/location |
| `destination_raw` | object | Raw destination string from AIS message |
| `eta` | datetime64[ns, UTC] | Estimated Time of Arrival |
| `nav_status` | object | Navigation status (e.g., "under way using engine") |
| `heading` | int64 | Ship's heading in degrees (0-359) |
| `lat` | float64 | Current latitude position |
| `lon` | float64 | Current longitude position |
| `delta_distance` | float64 | Distance traveled since last position (meters) |
| `delta_time` | float64 | Time elapsed since last position (hours) |
| `est_speed` | float64 | Estimated speed calculated from position changes |
| `true_destination_array` | object | Processed destination with timestamp |

### Key Characteristics

#### Temporal Features
- **Primary Time Index**: `mdt` (Message DateTime)
- **Prediction Target Time**: `next_mdt` 
- **Time Range**: 2018-2025
- **Time Zone**: UTC standardized

#### Spatial Features
- **Geographic Coverage**: Cape Town area (South Africa)
- **Coordinate System**: WGS84 decimal degrees
- **Position Accuracy**: High precision (6+ decimal places)

#### Movement Features
- **Speed Measurements**: Both reported (`speed`) and calculated (`est_speed`)
- **Distance Tracking**: `delta_distance` for trajectory analysis
- **Directional Data**: `heading` for ship orientation

#### Ship Identification
- **Unique Identifier**: `imo` number for ship tracking
- **Static Characteristics**: `draught` (ship loading condition)

## Data Quality Observations

### Strengths
- **High Volume**: Millions of records per year
- **Rich Feature Set**: 18 comprehensive attributes
- **Temporal Consistency**: UTC timestamps for all time fields
- **Calculated Fields**: Pre-computed `delta_distance`, `delta_time`, `est_speed`

### Potential Issues to Investigate
- **Missing Values**: Need to analyze null patterns across all files
- **Data Gaps**: 2021-2024 missing from dataset
- **Coordinate Precision**: Verify geographic bounds for Cape Town area
- **Speed Discrepancies**: Compare `speed` vs `est_speed` for accuracy
- **Destination Parsing**: Check consistency between `destination` and `destination_raw`

## Forecasting Implications

### Time Series Structure
- **Group Identifier**: `imo` (each ship is a separate time series)
- **Time Index**: `mdt` for chronological ordering
- **Prediction Target**: Various options:
  - Position forecasting: `lat`, `lon`
  - Speed prediction: `speed` or `est_speed`
  - ETA prediction: `eta`
  - Destination prediction: `destination`

### Feature Engineering Opportunities
- **Lag Features**: Historical positions, speeds, headings
- **Geographic Features**: Distance to ports, coastal proximity
- **Temporal Features**: Hour of day, day of week, seasonal patterns
- **Ship-specific Features**: Historical behavior patterns per `imo`
- **Route Features**: Common shipping lanes, port-to-port routes

### Model Architecture Considerations
- **Multi-variate Time Series**: Multiple features per timestamp
- **Variable Length Sequences**: Ships enter/exit the area
- **Irregular Sampling**: Non-uniform time intervals between messages
- **Hierarchical Structure**: Ship-level and fleet-level patterns

## Recommended Next Steps

### Data Preprocessing
1. **Validation**: Check all years have consistent schema
2. **Cleaning**: Handle missing values and outliers
3. **Geographic Filtering**: Ensure data bounds match Cape Town area
4. **Time Series Preparation**: Create proper time series format per ship

### Feature Engineering
1. **Derived Features**: Calculate additional movement metrics
2. **Geographic Features**: Add port distances, coastal features
3. **Temporal Features**: Extract time-based patterns
4. **Lag Features**: Create historical context windows

### Exploratory Analysis
1. **Traffic Patterns**: Analyze ship movement flows
2. **Seasonal Trends**: Identify temporal patterns
3. **Port Analysis**: Study arrival/departure patterns
4. **Speed Profiles**: Analyze speed distributions and changes

### Model Development
1. **Baseline Models**: Simple persistence and linear models
2. **Traditional ML**: Random Forest, XGBoost for tabular prediction
3. **Deep Learning**: LSTM, GRU, Transformer for sequence modeling
4. **Specialized Models**: TFT (Temporal Fusion Transformer), N-BEATS

## Technical Notes

### Data Loading
- Use `pickle.load()` for reading files
- Files are pandas DataFrames ready for analysis
- Consider chunking for memory management with large datasets

### Coordinate System
- Latitude/Longitude in decimal degrees
- Negative latitude indicates Southern Hemisphere (Cape Town area)
- Longitude around 18°E for Cape Town region

### Time Handling
- All datetime fields are UTC timezone-aware
- `mdt` represents the message timestamp
- `next_mdt` can be used for prediction target timing
- `eta` provides destination arrival estimates

---

*Generated by AIS Data Investigation Script on June 16, 2025*
