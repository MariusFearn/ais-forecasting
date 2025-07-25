"""
Data Investigation Script for AIS Forecasting Project

This script examines the AIS data files to understand their structure,
schema, and characteristics for building the forecasting models.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def load_ais_data(file_path):
    """
    Load AIS data from pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded data (DataFrame or other structure)
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded: {file_path}")
        return data
    except Exception as e:
        print(f"✗ Error loading {file_path}: {e}")
        return None

def examine_dataframe(df, name="DataFrame"):
    """
    Examine a DataFrame and print detailed information.
    
    Args:
        df: pandas DataFrame to examine
        name: Name for the DataFrame (for printing)
    """
    print(f"\n{'='*60}")
    print(f"EXAMINING {name.upper()}")
    print(f"{'='*60}")
    
    # Basic info
    print(f"\n📊 BASIC INFORMATION:")
    print(f"   Shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column information
    print(f"\n📋 COLUMNS ({len(df.columns)} total):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        print(f"   {i:2d}. {col:<25} | {str(dtype):<12} | Nulls: {null_count:>6} ({null_pct:5.1f}%) | Unique: {unique_count:>8}")
    
    # Data types summary
    print(f"\n🔢 DATA TYPES SUMMARY:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {str(dtype):<15}: {count:>3} columns")
    
    # Missing values
    print(f"\n❓ MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_data = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_data) > 0:
        print(missing_data.to_string(index=False))
    else:
        print("   No missing values found! 🎉")
    
    # Sample data
    print(f"\n📋 SAMPLE DATA (first 3 rows):")
    print(df.head(3).to_string())
    
    # Statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 STATISTICAL SUMMARY (numeric columns):")
        print(df[numeric_cols].describe().to_string())
    
    # Date/time columns analysis
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    object_cols = df.select_dtypes(include=['object']).columns
    
    # Check if any object columns might be dates
    potential_date_cols = []
    for col in object_cols:
        sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if sample_val and isinstance(sample_val, str):
            # Simple check for date-like strings
            if any(char in sample_val for char in ['-', '/', ':']):
                potential_date_cols.append(col)
    
    if len(datetime_cols) > 0 or len(potential_date_cols) > 0:
        print(f"\n📅 DATE/TIME ANALYSIS:")
        for col in datetime_cols:
            print(f"   {col}: {df[col].min()} to {df[col].max()}")
        for col in potential_date_cols:
            print(f"   {col} (potential date): {df[col].iloc[0]} (sample)")

def examine_data_structure(data, name="Data"):
    """
    Examine the structure of loaded data.
    
    Args:
        data: The loaded data
        name: Name for the data (for printing)
    """
    print(f"\n{'='*60}")
    print(f"DATA STRUCTURE ANALYSIS: {name}")
    print(f"{'='*60}")
    
    print(f"Type: {type(data)}")
    
    if isinstance(data, pd.DataFrame):
        examine_dataframe(data, name)
    elif isinstance(data, dict):
        print(f"\nDictionary with {len(data)} keys:")
        for key, value in data.items():
            print(f"\n🔑 Key: '{key}'")
            print(f"   Type: {type(value)}")
            if isinstance(value, pd.DataFrame):
                print(f"   DataFrame shape: {value.shape}")
                print(f"   Columns: {list(value.columns)}")
                examine_dataframe(value, f"{name}['{key}']")
            elif isinstance(value, (list, tuple)):
                print(f"   Length: {len(value)}")
                if len(value) > 0:
                    print(f"   Sample element type: {type(value[0])}")
            else:
                print(f"   Value: {str(value)[:100]}...")
    elif isinstance(data, (list, tuple)):
        print(f"\n{type(data).__name__} with {len(data)} elements")
        if len(data) > 0:
            print(f"First element type: {type(data[0])}")
            if isinstance(data[0], pd.DataFrame):
                examine_dataframe(data[0], f"{name}[0]")
    else:
        print(f"\nData content (first 500 chars):\n{str(data)[:500]}...")

def investigate_all_files():
    """
    Investigate all AIS pickle files in the raw_data directory.
    """
    raw_data_dir = Path("raw_data")
    
    if not raw_data_dir.exists():
        print(f"❌ Raw data directory not found: {raw_data_dir}")
        return
    
    # Find all pickle files
    pickle_files = list(raw_data_dir.glob("*.pkl"))
    
    if not pickle_files:
        print(f"❌ No pickle files found in: {raw_data_dir}")
        return
    
    print(f"🔍 Found {len(pickle_files)} pickle files:")
    for file in pickle_files:
        print(f"   - {file.name}")
    
    # Examine each file
    for file_path in pickle_files:
        print(f"\n\n{'='*80}")
        print(f"INVESTIGATING: {file_path.name}")
        print(f"{'='*80}")
        
        data = load_ais_data(file_path)
        if data is not None:
            examine_data_structure(data, file_path.name)

def investigate_specific_file(filename):
    """
    Investigate a specific AIS file.
    
    Args:
        filename: Name of the file to investigate (e.g., "ais_cape_data_2018.pkl")
    """
    file_path = Path("raw_data") / filename
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    print(f"🔍 Investigating: {filename}")
    data = load_ais_data(file_path)
    if data is not None:
        examine_data_structure(data, filename)
    
    return data

def quick_comparison():
    """
    Quick comparison of all available years.
    """
    print(f"\n{'='*60}")
    print("QUICK COMPARISON OF ALL YEARS")
    print(f"{'='*60}")
    
    raw_data_dir = Path("raw_data")
    pickle_files = sorted(raw_data_dir.glob("*.pkl"))
    
    summary_data = []
    
    for file_path in pickle_files:
        data = load_ais_data(file_path)
        if data is not None and isinstance(data, pd.DataFrame):
            summary_data.append({
                'File': file_path.name,
                'Rows': len(data),
                'Columns': len(data.columns),
                'Memory (MB)': data.memory_usage(deep=True).sum() / 1024**2,
                'Date Range': f"{data.index.min()} to {data.index.max()}" if hasattr(data.index, 'min') else "N/A"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

def main():
    """
    Main function to run the data investigation.
    """
    print("🚢 AIS Data Investigation Tool")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    # Check if we're in the right place
    if not Path("raw_data").exists():
        print("❌ raw_data directory not found. Make sure you're in the project root.")
        return
    
    print("\nChoose an option:")
    print("1. Investigate all files")
    print("2. Investigate 2018 data")
    print("3. Investigate 2019 data") 
    print("4. Investigate 2020 data")
    print("5. Investigate 2025 data")
    print("6. Quick comparison")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        investigate_all_files()
    elif choice == "2":
        investigate_specific_file("ais_cape_data_2018.pkl")
    elif choice == "3":
        investigate_specific_file("ais_cape_data_2019.pkl")
    elif choice == "4":
        investigate_specific_file("ais_cape_data_2020.pkl")
    elif choice == "5":
        investigate_specific_file("ais_cape_data_2025.pkl")
    elif choice == "6":
        quick_comparison()
    else:
        print("Invalid choice. Investigating 2018 data by default...")
        investigate_specific_file("ais_cape_data_2018.pkl")

if __name__ == "__main__":
    main()
