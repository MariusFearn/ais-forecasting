# Raw Data Directory

This directory contains the original, unprocessed AIS data files.

## Expected Data Format

- **File formats**: CSV, Parquet
- **Required columns**:
  - `timestamp`: Datetime of the AIS message
  - `vessel_id`: Unique identifier for the vessel
  - `lat`: Latitude coordinate
  - `lon`: Longitude coordinate
  - `speed`: Vessel speed (optional)
  - `course`: Vessel course/heading (optional)

## Data Organization

Organize files by:
- Date ranges (e.g., `2024-01/`, `2024-02/`)
- Geographic regions
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
