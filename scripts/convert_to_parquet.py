"""Convert AIS pickle files to Parquet for DuckDB optimization."""
import pandas as pd
from pathlib import Path
import logging
import numpy as np

def convert_pickle_to_parquet():
    """Convert all pickle files to optimized Parquet format."""
    
    raw_data_dir = Path("data/raw")
    output_dir = raw_data_dir / "parquet"
    output_dir.mkdir(exist_ok=True)
    
    pickle_files = list(raw_data_dir.glob("ais_cape_data_*.pkl"))
    
    print(f"🔄 Converting {len(pickle_files)} pickle files to Parquet...")
    
    for pickle_file in pickle_files:
        parquet_file = output_dir / pickle_file.with_suffix('.parquet').name
        
        if parquet_file.exists():
            print(f"   ⏭️  Skipping {pickle_file.name} (already exists)")
            continue
            
        print(f"   📦 Converting {pickle_file.name}...")
        
        # Load and optimize
        df = pd.read_pickle(pickle_file)
        
        # Optimize data types before saving
        df = optimize_dtypes(df)
        
        # Save as Parquet with optimal compression
        df.to_parquet(
            parquet_file,
            compression='snappy',  # Fast compression
            index=False
        )
        
        # Report size reduction
        original_size = pickle_file.stat().st_size / 1024**2  # MB
        new_size = parquet_file.stat().st_size / 1024**2  # MB
        reduction = (original_size - new_size) / original_size * 100
        
        print(f"      ✅ {original_size:.1f}MB → {new_size:.1f}MB ({reduction:.1f}% reduction)")

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize data types for better Parquet compression."""
    
    # Convert timestamps
    datetime_cols = ['mdt', 'next_mdt', 'eta']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Optimize numeric types
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert string columns with low cardinality to category
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    return df

if __name__ == "__main__":
    convert_pickle_to_parquet()
