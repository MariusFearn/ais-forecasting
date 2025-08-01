#!/usr/bin/env python3
"""
Data Loading Benchmark Script

This script benchmarks the performance of loading AIS data using two different backends:
1.  Pandas reading multiple Parquet files.
2.  The new DuckDB-based loader querying the same Parquet files.

It measures the average time taken over several runs to provide a clear comparison
of the performance gains from the DuckDB implementation.

Usage:
    python scripts/benchmark_data_loading.py --config creation_data_comprehensive --runs 3
"""

import time
import sys
from pathlib import Path
import yaml
import pandas as pd
import re
import argparse
import os

# Add src to path for local imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.loader import AISDataLoader

def load_config(config_name):
    """
    Loads a specified experiment config by recursively merging it with its defaults.
    This is the same robust loader used by the main training scripts.
    """
    config_path = Path(f"config/experiment_configs/{config_name}.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    def merge_configs(base, override):
        """Deep merge two configurations."""
        if base is None: return override
        if override is None: return base
        if not isinstance(base, dict) or not isinstance(override, dict): return override
        
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def load_config_recursive(path, visited=None):
        """Recursively load configurations with defaults."""
        if visited is None: visited = set()
        if str(path) in visited: return {}
        visited.add(str(path))
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'defaults' not in config:
            return config
        
        result = {}
        for default_file in config['defaults']:
            # Resolve path relative to the current config file
            base_path = path.parent
            if default_file.startswith('../'):
                default_path = (base_path.parent / f"{default_file[3:]}.yaml").resolve()
            else:
                default_path = (base_path / f"{default_file}.yaml").resolve()
            
            if default_path.exists():
                default_config = load_config_recursive(default_path, visited.copy())
                result = merge_configs(result, default_config)
        
        current_config = {k: v for k, v in config.items() if k != 'defaults'}
        return merge_configs(result, current_config)
    
    return load_config_recursive(config_path)

def get_years_from_config(config):
    """Extracts year strings from data file paths in the config."""
    data_files = config['data_source']['data_files']
    return sorted(list(set(re.findall(r'(\d{4})', " ".join(data_files)))))

def benchmark_pandas(data_dir, years, runs):
    """Benchmarks data loading using Pandas to read multiple Parquet files."""
    total_time = 0
    
    print("   Performing warm-up run for Pandas...")
    all_data = []
    for year in years:
        file_path = os.path.join(data_dir, "raw", "parquet", f"ais_cape_data_{year}.parquet")
        if os.path.exists(file_path):
            all_data.append(pd.read_parquet(file_path))
    if all_data:
        pd.concat(all_data, ignore_index=True)
    print("   Warm-up complete.")

    for i in range(runs):
        print(f"   Running trial {i + 1}/{runs}...")
        start_time = time.time()
        
        all_data = []
        for year in years:
            file_path = os.path.join(data_dir, "raw", "parquet", f"ais_cape_data_{year}.parquet")
            if os.path.exists(file_path):
                all_data.append(pd.read_parquet(file_path))
        
        if not all_data:
            print("      No data files found for Pandas benchmark. Skipping.")
            return 0

        df = pd.concat(all_data, ignore_index=True)

        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time
        print(f"      Trial {i + 1} time: {run_time:.4f}s, Loaded {len(df):,} records")
        
    return total_time / runs if runs > 0 else 0

def benchmark_duckdb(data_dir, years, runs):
    """Benchmarks data loading using the DuckDB-powered AISDataLoader."""
    total_time = 0
    loader = AISDataLoader(data_dir=data_dir, use_duckdb=True)
    
    print("   Performing warm-up run for DuckDB...")
    loader.load_multi_year_data_optimized(years)
    print("   Warm-up complete.")

    for i in range(runs):
        print(f"   Running trial {i + 1}/{runs}...")
        start_time = time.time()
        df = loader.load_multi_year_data_optimized(years)
        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time
        print(f"      Trial {i + 1} time: {run_time:.4f}s, Loaded {len(df):,} records")
        
    return total_time / runs if runs > 0 else 0

def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description='Benchmark data loading performance.')
    parser.add_argument('--config', default='creation_data_comprehensive',
                        help='Data creation configuration name to use for benchmarking.')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of timed runs to perform for each method.')
    
    args = parser.parse_args()

    print(f"🚀 Starting benchmark using '{args.config}' configuration...")
    print(f"   Running each method {args.runs} times...")

    config = load_config(args.config)
    
    data_files = config.get('data_source', {}).get('data_files')
    if not data_files:
        raise ValueError("Config must contain 'data_files' under 'data_source'.")
        
    data_dir = str(Path(data_files[0]).resolve().parent.parent)
    years = get_years_from_config(config)
    
    print(f"   Data directory inferred as: {data_dir}")
    
    # --- Benchmark Pandas ---
    print("
🐼 Benchmarking Pandas reading Parquet files...")
    pandas_avg_time = benchmark_pandas(data_dir, years, runs=args.runs)
    print(f"   ✅ Average Pandas (reading parquet) time: {pandas_avg_time:.4f} seconds")

    # --- Benchmark DuckDB ---
    print("
🦆 Benchmarking new DuckDB loader (use_duckdb=True)...")
    duckdb_avg_time = benchmark_duckdb(data_dir, years, runs=args.runs)
    print(f"   ✅ Average DuckDB time: {duckdb_avg_time:.4f} seconds")

    # --- Print Final Results ---
    print("
" + "="*30)
    print("    Benchmark Results Summary")
    print("="*30)
    print(f"Configuration:      {args.config}")
    print(f"Runs per method:    {args.runs}")
    print(f"Years loaded:       {', '.join(years)}")
    print("-" * 30)
    print(f"Pandas (Parquet):   {pandas_avg_time:.4f}s")
    print(f"DuckDB (Parquet):   {duckdb_avg_time:.4f}s")
    print("-" * 30)
    
    if duckdb_avg_time > 0 and pandas_avg_time > 0:
        speedup = pandas_avg_time / duckdb_avg_time
        print(f"
🎉 DuckDB is {speedup:.2f}x faster than Pandas for this workload!")
    else:
        print("
⚠️  Could not calculate speedup due to zero or negative timing.")

if __name__ == "__main__":
    main()

import time
import sys
from pathlib import Path
import yaml
import pandas as pd
import re
import argparse
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.loader import AISDataLoader

def load_config(config_name):
    """Loads a specified experiment config by merging it with the default config."""
    default_config_path = Path("config/default.yaml")
    exp_config_path = Path(f"config/experiment_configs/{config_name}.yaml")

    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config not found at: {default_config_path}")
    if not exp_config_path.exists():
        raise FileNotFoundError(f"Experiment config not found at: {exp_config_path}")

    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(exp_config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
        
    # Deep merge experiment config into default config
    def merge_configs(base, override):
        if base is None: return override
        if override is None: return base
        if not isinstance(base, dict) or not isinstance(override, dict): return override
        
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    return merge_configs(config, exp_config)

def get_years_from_config(config):
    """Extracts year strings from data file paths in the config."""
    data_files = config['data_source']['data_files']
    return sorted(list(set(re.findall(r'(\d{4})', " ".join(data_files)))))

def benchmark_pandas(data_dir, years, runs):
    """Benchmarks data loading using Pandas to read Parquet files."""
    total_time = 0
    
    # Perform a warm-up run
    print("   Performing warm-up run for Pandas...")
    all_data = []
    for year in years:
        file_path = os.path.join(data_dir, "raw", "parquet", f"ais_cape_data_{year}.parquet")
        if os.path.exists(file_path):
            all_data.append(pd.read_parquet(file_path))
    if all_data:
        pd.concat(all_data, ignore_index=True)
    print("   Warm-up complete.")

    for i in range(runs):
        print(f"   Running trial {i + 1}/{runs}...")
        start_time = time.time()
        
        all_data = []
        for year in years:
            file_path = os.path.join(data_dir, "raw", "parquet", f"ais_cape_data_{year}.parquet")
            if os.path.exists(file_path):
                all_data.append(pd.read_parquet(file_path))
        
        if not all_data:
            print("      No data files found for Pandas benchmark. Skipping.")
            return 0

        df = pd.concat(all_data, ignore_index=True)

        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time
        print(f"      Trial {i + 1} time: {run_time:.4f}s, Loaded {len(df):,} records")
        
    return total_time / runs

def benchmark_duckdb(data_dir, years, runs):
    """Benchmarks data loading using the DuckDB-powered AISDataLoader."""
    total_time = 0
    loader = AISDataLoader(data_dir=data_dir, use_duckdb=True)
    
    print("   Performing warm-up run for DuckDB...")
    loader.load_multi_year_data_optimized(years)
    print("   Warm-up complete.")

    for i in range(runs):
        print(f"   Running trial {i + 1}/{runs}...")
        start_time = time.time()
        df = loader.load_multi_year_data_optimized(years)
        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time
        print(f"      Trial {i + 1} time: {run_time:.4f}s, Loaded {len(df):,} records")
        
    return total_time / runs

def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description='Benchmark data loading performance.')
    parser.add_argument('--config', default='experiment_h3_comprehensive',
                        help='Configuration name to use for benchmarking.')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of timed runs to perform for each method.')
    
    args = parser.parse_args()

    print(f"🚀 Starting benchmark using '{args.config}' configuration...")
    print(f"   Running each method {args.runs} times...")

    config = load_config(args.config)
    
    # Infer data_dir from the data_files list, which is robust
    data_files = config['data_source']['data_files']
    if not data_files:
        raise ValueError("Config must contain 'data_files' under 'data_source'.")
    data_dir = str(Path(data_files[0]).resolve().parent.parent)
    years = get_years_from_config(config)
    
    print(f"   Data directory inferred as: {data_dir}")
    
    # --- Benchmark Pandas ---
    print("\n🐼 Benchmarking Pandas reading Parquet files...")
    pandas_avg_time = benchmark_pandas(data_dir, years, runs=args.runs)
    print(f"   ✅ Average Pandas (reading parquet) time: {pandas_avg_time:.4f} seconds")

    # --- Benchmark DuckDB ---
    print("\n🦆 Benchmarking new DuckDB loader (use_duckdb=True)...")
    duckdb_avg_time = benchmark_duckdb(data_dir, years, runs=args.runs)
    print(f"   ✅ Average DuckDB time: {duckdb_avg_time:.4f} seconds")

    # --- Print Final Results ---
    print("\n" + "="*30)
    print("    Benchmark Results Summary")
    print("="*30)
    print(f"Configuration:      {args.config}")
    print(f"Runs per method:    {args.runs}")
    print(f"Years loaded:       {', '.join(years)}")
    print("-" * 30)
    print(f"Pandas (Parquet):   {pandas_avg_time:.4f}s")
    print(f"DuckDB (Parquet):   {duckdb_avg_time:.4f}s")
    print("-" * 30)
    
    if duckdb_avg_time > 0 and pandas_avg_time > 0:
        speedup = pandas_avg_time / duckdb_avg_time
        print(f"\n🎉 DuckDB is {speedup:.2f}x faster than Pandas for this workload!")
    else:
        print("\n⚠️  Could not calculate speedup due to zero or negative timing.")

if __name__ == "__main__":
    main()
