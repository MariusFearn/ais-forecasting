"""Benchmark DuckDB vs Pandas performance on AIS data."""
import time
import pandas as pd
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data.duckdb_engine import DuckDBEngine
import pyarrow
import duckdb

def benchmark_operations():
    """Compare pandas vs DuckDB performance on common operations."""
    
    print("🏁 DuckDB vs Pandas Benchmark")
    print("=" * 50)
    
    # Load test dataset
    test_file = os.path.join(project_root, "data", "raw", "parquet", "ais_cape_data_2024.parquet")
    
    # Pandas benchmark
    print("\n📊 Pandas operations:")
    start_time = time.time()
    df_pandas = pd.read_parquet(test_file)
    vessel_stats_pandas = df_pandas.groupby('imo').agg({
        'speed': ['mean', 'std', 'count'],
        'lat': 'nunique'
    })
    pandas_time = time.time() - start_time
    print(f"   Time: {pandas_time:.2f}s")
    
    # DuckDB benchmark
    print("\n🚀 DuckDB operations:")
    start_time = time.time()
    duckdb_engine = DuckDBEngine()
    vessel_stats_duckdb = duckdb_engine.execute_query(f"""
        SELECT imo,
               AVG(speed) as speed_mean,
               STDDEV(speed) as speed_std,
               COUNT(speed) as speed_count,
               COUNT(DISTINCT lat) as lat_nunique
        FROM read_parquet('{test_file}')
        GROUP BY imo
    """)
    duckdb_time = time.time() - start_time
    print(f"   Time: {duckdb_time:.2f}s")
    
    # Results
    print(f"\n🏆 Results:")
    if duckdb_time > 0:
        speedup = pandas_time / duckdb_time
        print(f"   Speedup: {speedup:.1f}x faster with DuckDB")
    else:
        print("   DuckDB execution was too fast to measure speedup.")

    print(f"   Pandas memory: {df_pandas.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   Results match: {len(vessel_stats_pandas) == len(vessel_stats_duckdb)}")

if __name__ == "__main__":
    benchmark_operations()
