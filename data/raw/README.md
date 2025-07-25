# Raw Data Directory

This directory contains the original, unprocessed AIS data files for Cape Town maritime area.

## Current Data Files

- `ais_cape_data_2018.pkl` - AIS data for 2018 (8.3M+ records)
- `ais_cape_data_2019.pkl` - AIS data for 2019
- `ais_cape_data_2020.pkl` - AIS data for 2020
- `ais_cape_data_2021.pkl` - AIS data for 2021
- `ais_cape_data_2022.pkl` - AIS data for 2022
- `ais_cape_data_2023.pkl` - AIS data for 2023
- `ais_cape_data_2024.pkl` - AIS data for 2024
- `ais_cape_data_2025.pkl` - AIS data for 2025

## Data Format

- **File format**: Pickle files containing pandas DataFrames
- **Schema**: 18 columns per record including:
  - `mdt`: Message DateTime (primary time index)
  - `imo`: International Maritime Organization number (vessel ID)
  - `lat`, `lon`: Current position coordinates
  - `lat1`, `lon1`: Previous position coordinates
  - `speed`: Speed over ground (knots)
  - `heading`: Ship's heading (degrees)
  - `draught`: Ship's draught depth (meters)
  - `nav_status`: Navigation status
  - `destination`: Processed destination
  - `eta`: Estimated Time of Arrival
  - Additional calculated fields (delta_distance, delta_time, etc.)

## Data Organization

Files organized by year covering Cape Town maritime region.
- Data sources

## Example Files

```
raw/
├── 2024-01/
│   ├── ais_data_2024-01-01.csv
│   ├── ais_data_2024-01-02.csv
│   └── ...
├── 2024-02/
│   └── ...
└── vessel_metadata.csv
```
