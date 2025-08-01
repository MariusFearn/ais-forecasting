"""Global maritime data loading leveraging existing DuckDB optimization."""

from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import logging
import time

from .duckdb_engine import DuckDBEngine
from .loader import AISDataLoader
from ..utils.logging_setup import get_logger, log_performance_metrics

logger = get_logger(__name__)

def load_global_ais_data(
    years: List[str] = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    filters: Optional[Dict] = None,
    use_duckdb: bool = True
) -> pd.DataFrame:
    """
    Load global AIS data using existing optimized DuckDB engine.
    
    Uses original column names: imo, mdt, lat, lon (no renaming needed)
    Leverages parquet files for 10-50x speedup over pickle files.
    
    Args:
        years: List of years to load (e.g., ["2018", "2019"])
        filters: Optional filtering criteria for DuckDB
        use_duckdb: Whether to use DuckDB optimization (recommended)
        
    Returns:
        DataFrame with columns: imo, mdt, lat, lon
        
    Raises:
        FileNotFoundError: If parquet files don't exist
        ValueError: If no data is loaded
    """
    logger.info(f"Loading global AIS data for years: {years}")
    start_time = time.time()
    
    try:
        # Use existing optimized AISDataLoader
        loader = AISDataLoader(data_dir="data", use_duckdb=use_duckdb)
        
        if use_duckdb:
            # Use DuckDB for ultra-fast loading
            logger.info("Using DuckDB engine for optimized data loading")
            ais_data = loader.load_multi_year_data_optimized(years, filters=filters)
        else:
            # Fallback to standard loading
            logger.info("Using standard data loading")
            ais_data = loader.load_multi_year_data_original(years, **filters if filters else {})
        
        if ais_data.empty:
            raise ValueError(f"No data loaded for years: {years}")
        
        # Log performance metrics
        duration = time.time() - start_time
        log_performance_metrics(
            logger, 
            "Global AIS Data Loading", 
            duration, 
            len(ais_data)
        )
        
        logger.info(f"Successfully loaded {len(ais_data):,} AIS records "
                   f"from {ais_data['imo'].nunique():,} unique vessels")
        
        return ais_data
        
    except Exception as e:
        logger.error(f"Failed to load global AIS data: {e}")
        raise

def validate_ais_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate AIS data quality using original column names.
    
    Args:
        df: AIS DataFrame with columns [imo, mdt, lat, lon]
        
    Returns:
        Validated and cleaned DataFrame
        
    Raises:
        ValueError: If critical validation checks fail
    """
    logger.debug("Validating AIS data quality")
    
    original_size = len(df)
    
    # Check required columns (original names)
    required_columns = ['imo', 'mdt', 'lat', 'lon']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove rows with invalid coordinates
    invalid_coords = (
        (df['lat'].abs() > 90) | 
        (df['lon'].abs() > 180) |
        df['lat'].isna() |
        df['lon'].isna()
    )
    df = df[~invalid_coords].copy()
    
    # Remove rows with invalid IMO
    df = df[df['imo'].notna() & (df['imo'] > 0)].copy()
    
    # Remove rows with invalid timestamps
    df = df[df['mdt'].notna()].copy()
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['mdt']):
        df['mdt'] = pd.to_datetime(df['mdt'], errors='coerce')
        df = df[df['mdt'].notna()].copy()
    
    # Log validation results
    removed_count = original_size - len(df)
    if removed_count > 0:
        removal_rate = removed_count / original_size * 100
        logger.warning(f"Removed {removed_count:,} invalid records ({removal_rate:.1f}%)")
    
    logger.debug(f"Data validation complete: {len(df):,} valid records")
    return df

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive summary statistics for AIS dataset.
    
    Args:
        df: AIS DataFrame with original column names
        
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'total_records': len(df),
        'unique_vessels': df['imo'].nunique(),
        'date_range': {
            'start': df['mdt'].min(),
            'end': df['mdt'].max()
        },
        'geographic_bounds': {
            'lat_min': df['lat'].min(),
            'lat_max': df['lat'].max(),
            'lon_min': df['lon'].min(),
            'lon_max': df['lon'].max()
        },
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    return summary
