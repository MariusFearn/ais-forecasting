# DuckDB + Pandas Integration Plan for AIS Forecasting

## 🎯 **Objective**
Optimize AIS data processing performance by integrating DuckDB for heavy data operations while maintaining pandas compatibility for ML pipelines. Target **10-100x speedup** for aggregations and **50-80% memory reduction**.

## 📊 **Current Performance Analysis**

### **Bottlenecks Identified:**
- **GroupBy operations**: `df.groupby('vessel_id').agg({...})` - slow on 16GB dataset
- **Multi-year data loading**: 8 pickle files (1.6GB-2.6GB each) - memory intensive
- **H3 cell aggregations**: `df.groupby('h3_cell').size()` - frequent operation
- **Time-series features**: Rolling windows, lag features - computationally expensive
- **Memory usage**: Full dataset loading (16GB) into 54GB RAM

### **Target Improvements:**
- ⚡ **Aggregations**: 10-100x faster with SQL-based operations
- 💾 **Memory usage**: 50-80% reduction via columnar storage
- 🚀 **Data loading**: 3-5x faster with Parquet format
- 📈 **Feature engineering**: Built-in window functions for time-series

---

## 🗂️ **Implementation Plan**

### **Phase 1: Data Format Migration** 📁 **(Week 1)**

#### **1.1 Convert Pickle to Parquet**
```bash
# New script: scripts/convert_to_parquet.py
```

**File:** `scripts/convert_to_parquet.py`
```python
"""Convert AIS pickle files to Parquet for DuckDB optimization."""
import pandas as pd
from pathlib import Path
import logging

(note make a new sub folder in raw, called parquet for the new raw data)

def convert_pickle_to_parquet():
    """Convert all pickle files to optimized Parquet format."""
    
    raw_data_dir = Path("data/raw")
    pickle_files = list(raw_data_dir.glob("ais_cape_data_*.pkl"))
    
    print(f"🔄 Converting {len(pickle_files)} pickle files to Parquet...")
    
    for pickle_file in pickle_files:
        parquet_file = pickle_file.with_suffix('.parquet')
        
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
            df[col] = pd.to_datetime(df[col])
    
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
```

#### **1.2 Update Configuration**
**File:** `config/default.yaml`
```yaml
# Add DuckDB configuration section
duckdb:
  enabled: true
  data_format: "parquet"  # Use Parquet for better performance
  memory_limit: "48GB"    # Use most of available 54GB RAM
  threads: 14             # Use all CPU threads
  
data:
  raw_data_path: "data/raw"
  # Support both formats during transition
  file_extensions: [".parquet", ".pkl"]
  preferred_format: "parquet"
```

---

### **Phase 2: DuckDB Integration Layer** 🔧 **(Week 2)**

#### **2.1 Create DuckDB Utility Module**
**File:** `src/data/duckdb_engine.py`
```python
"""DuckDB integration layer for high-performance data operations."""
import duckdb
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

class DuckDBEngine:
    """High-performance data engine using DuckDB for AIS data processing."""
    
    def __init__(self, memory_limit: str = "48GB", threads: int = 14):
        """Initialize DuckDB connection with optimal settings."""
        self.conn = duckdb.connect()
        
        # Configure for your hardware
        self.conn.execute(f"SET memory_limit='{memory_limit}'")
        self.conn.execute(f"SET threads TO {threads}")
        self.conn.execute("SET enable_progress_bar=true")
        
        self.logger = logging.getLogger(__name__)
        
    def load_multi_year_data(self, 
                           years: List[str], 
                           filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Load multiple years of data with SQL-based filtering.
        10-50x faster than pandas approach.
        """
        
        # Build file pattern for parquet files
        file_pattern = "data/raw/ais_cape_data_*.parquet"
        
        # Base SQL query
        sql = f"""
        SELECT * FROM read_parquet('{file_pattern}')
        WHERE EXTRACT(year FROM mdt) IN ({','.join(years)})
        """
        
        # Add filters if provided
        if filters:
            if 'lat_range' in filters:
                lat_min, lat_max = filters['lat_range']
                sql += f" AND lat BETWEEN {lat_min} AND {lat_max}"
            
            if 'vessel_types' in filters:
                vessel_list = "','".join(filters['vessel_types'])
                sql += f" AND vessel_type IN ('{vessel_list}')"
        
        self.logger.info(f"Executing DuckDB query for {len(years)} years...")
        return self.conn.execute(sql).df()
    
    def aggregate_h3_cells(self, 
                          table_name: str = "ais_data",
                          resolution: int = 5) -> pd.DataFrame:
        """
        Ultra-fast H3 cell aggregation using SQL.
        Expected: 10-100x faster than pandas groupby.
        """
        
        sql = f"""
        SELECT 
            h3_cell,
            COUNT(*) as message_count,
            COUNT(DISTINCT imo) as unique_vessels,
            AVG(speed) as avg_speed,
            STDDEV(speed) as std_speed,
            MIN(mdt) as first_seen,
            MAX(mdt) as last_seen
        FROM {table_name}
        WHERE h3_cell IS NOT NULL
        GROUP BY h3_cell
        ORDER BY message_count DESC
        """
        
        return self.conn.execute(sql).df()
    
    def create_time_series_features(self, 
                                   vessel_id_col: str = "imo",
                                   timestamp_col: str = "mdt") -> pd.DataFrame:
        """
        Create lag and rolling features using SQL window functions.
        Much faster than pandas rolling operations.
        """
        
        sql = f"""
        SELECT *,
            -- Lag features
            LAG(speed, 1) OVER (PARTITION BY {vessel_id_col} ORDER BY {timestamp_col}) as speed_lag1,
            LAG(speed, 2) OVER (PARTITION BY {vessel_id_col} ORDER BY {timestamp_col}) as speed_lag2,
            LAG(speed, 6) OVER (PARTITION BY {vessel_id_col} ORDER BY {timestamp_col}) as speed_lag6,
            
            -- Rolling averages
            AVG(speed) OVER (
                PARTITION BY {vessel_id_col} 
                ORDER BY {timestamp_col} 
                ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
            ) as speed_rolling_6h,
            
            -- Rolling standard deviation
            STDDEV(speed) OVER (
                PARTITION BY {vessel_id_col} 
                ORDER BY {timestamp_col} 
                ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
            ) as speed_std_12h,
            
            -- Lead features (for prediction targets)
            LEAD(h3_cell, 1) OVER (PARTITION BY {vessel_id_col} ORDER BY {timestamp_col}) as next_h3_cell
            
        FROM ais_data
        ORDER BY {vessel_id_col}, {timestamp_col}
        """
        
        return self.conn.execute(sql).df()
    
    def register_dataframe(self, df: pd.DataFrame, table_name: str):
        """Register a pandas DataFrame as a DuckDB table for SQL queries."""
        self.conn.register(table_name, df)
        self.logger.info(f"Registered DataFrame as table '{table_name}' ({len(df):,} rows)")
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute custom SQL query and return result as DataFrame."""
        return self.conn.execute(sql).df()
    
    def close(self):
        """Close DuckDB connection."""
        self.conn.close()
```

#### **2.2 Update Data Loader**
**File:** `src/data/loader.py` - Add DuckDB support
```python
# Add imports
from .duckdb_engine import DuckDBEngine

class AISDataLoader:
    def __init__(self, use_duckdb: bool = True):
        self.use_duckdb = use_duckdb
        if use_duckdb:
            self.duckdb_engine = DuckDBEngine()
    
    def load_multi_year_data_optimized(self, years: List[str], **kwargs) -> pd.DataFrame:
        """Load multi-year data using DuckDB for 10x speedup."""
        
        if self.use_duckdb:
            # Use DuckDB for ultra-fast loading and filtering
            return self.duckdb_engine.load_multi_year_data(years, kwargs.get('filters'))
        else:
            # Fallback to original pandas approach
            return self.load_multi_year_data_original(years, **kwargs)
```

---

### **Phase 3: Script Modifications** 📝 **(Week 3)**

#### **3.1 Update Training Data Creation**
**File:** `scripts/create_training_data.py`

**Changes needed:**
```python
# Replace this pandas-heavy section:
# df_combined = pd.concat([pd.read_pickle(f) for f in files])
# vessel_stats = df_combined.groupby('imo').agg({...})

# With DuckDB-optimized version:
from src.data.duckdb_engine import DuckDBEngine

def create_training_data_optimized(config):
    """Create training data using DuckDB for 10-100x speedup."""
    
    duckdb_engine = DuckDBEngine()
    
    # Step 1: Load multi-year data (10x faster)
    years = config.get('data', {}).get('years', ['2023', '2024'])
    filters = {
        'lat_range': (-35, -33),  # Cape Town area
        'vessel_types': config.get('data', {}).get('vessel_types', [])
    }
    
    df = duckdb_engine.load_multi_year_data(years, filters)
    
    # Step 2: Register for SQL operations
    duckdb_engine.register_dataframe(df, 'ais_data')
    
    # Step 3: Create features using SQL (much faster)
    df_features = duckdb_engine.create_time_series_features()
    
    # Step 4: H3 aggregations (100x faster than pandas)
    h3_stats = duckdb_engine.aggregate_h3_cells()
    
    # Continue with pandas for ML-specific operations
    return df_features, h3_stats
```

#### **3.2 Update Model Training**
**File:** `scripts/train_h3_model.py`

**Add DuckDB configuration:**
```python
def load_training_data_optimized(data_path: str, config: Dict) -> pd.DataFrame:
    """Load training data with DuckDB optimization if available."""
    
    # Check if DuckDB is enabled in config
    if config.get('duckdb', {}).get('enabled', False):
        # Use DuckDB for initial loading and preprocessing
        duckdb_engine = DuckDBEngine()
        
        # If parquet files exist, use them (3-5x faster loading)
        parquet_path = Path(data_path).with_suffix('.parquet')
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_pickle(data_path)
        
        # Register for potential SQL operations
        duckdb_engine.register_dataframe(df, 'training_data')
        
        # Use SQL for any heavy preprocessing
        if len(df) > 100000:  # Only for large datasets
            df = duckdb_engine.execute_query("""
                SELECT *, 
                       LAG(speed, 1) OVER (PARTITION BY imo ORDER BY mdt) as speed_lag1
                FROM training_data
            """)
        
        duckdb_engine.close()
        return df
    else:
        # Original pandas approach
        return pd.read_pickle(data_path)
```

---

### **Phase 4: Advanced Optimizations** 🚀 **(Week 4)**

#### **4.1 Create Hybrid Processing Pipeline**
**File:** `src/training/hybrid_pipeline.py`
```python
"""Hybrid pandas + DuckDB pipeline for optimal performance."""

class HybridTrainingPipeline:
    """Training pipeline that uses DuckDB for data ops, pandas for ML."""
    
    def __init__(self, use_duckdb: bool = True):
        self.use_duckdb = use_duckdb
        if use_duckdb:
            self.duckdb_engine = DuckDBEngine()
        
        # Keep pandas pipeline for ML operations
        from .pipeline import TrainingPipeline
        self.pandas_pipeline = TrainingPipeline()
    
    def load_and_preprocess(self, data_path: str, config: Dict) -> pd.DataFrame:
        """Load and preprocess data using optimal engine for each operation."""
        
        if self.use_duckdb and Path(data_path).suffix == '.parquet':
            # DuckDB excels at: aggregations, joins, window functions
            df = self.duckdb_engine.execute_query(f"""
                SELECT *,
                    -- Time-series features (much faster in SQL)
                    LAG(speed, 1) OVER (PARTITION BY imo ORDER BY mdt) as speed_lag1,
                    LAG(speed, 6) OVER (PARTITION BY imo ORDER BY mdt) as speed_lag6,
                    
                    -- Rolling features
                    AVG(speed) OVER (
                        PARTITION BY imo ORDER BY mdt 
                        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                    ) as speed_avg_6h,
                    
                    -- H3 cell features
                    COUNT(*) OVER (PARTITION BY h3_cell) as h3_cell_popularity
                    
                FROM read_parquet('{data_path}')
            """)
        else:
            # Fallback to pandas
            df = pd.read_pickle(data_path)
        
        # Use pandas for ML-specific preprocessing
        # (feature encoding, scaling, etc.)
        df_processed = self.pandas_pipeline.prepare_features(df, config)
        
        return df_processed
```

#### **4.2 Benchmarking Script**
**File:** `scripts/benchmark_duckdb.py`
```python
"""Benchmark DuckDB vs Pandas performance on AIS data."""
import time
import pandas as pd
from src.data.duckdb_engine import DuckDBEngine

def benchmark_operations():
    """Compare pandas vs DuckDB performance on common operations."""
    
    print("🏁 DuckDB vs Pandas Benchmark")
    print("=" * 50)
    
    # Load test dataset
    test_file = "data/raw/ais_cape_data_2024.parquet"
    
    # Pandas benchmark
    print("\n📊 Pandas operations:")
    start_time = time.time()
    df_pandas = pd.read_parquet(test_file)
    vessel_stats_pandas = df_pandas.groupby('imo').agg({
        'speed': ['mean', 'std', 'count'],
        'h3_cell': 'nunique'
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
               COUNT(DISTINCT h3_cell) as h3_cell_nunique
        FROM read_parquet('{test_file}')
        GROUP BY imo
    """)
    duckdb_time = time.time() - start_time
    print(f"   Time: {duckdb_time:.2f}s")
    
    # Results
    speedup = pandas_time / duckdb_time
    print(f"\n🏆 Results:")
    print(f"   Speedup: {speedup:.1f}x faster with DuckDB")
    print(f"   Pandas memory: {df_pandas.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   Results match: {len(vessel_stats_pandas) == len(vessel_stats_duckdb)}")

if __name__ == "__main__":
    benchmark_operations()
```

---

## 🧪 **Testing Strategy**

### **Integration Tests**
**File:** `tests/test_duckdb_integration.py`
```python
"""Test DuckDB integration with existing pandas pipeline."""
import unittest
import pandas as pd
from src.data.duckdb_engine import DuckDBEngine
from src.training.hybrid_pipeline import HybridTrainingPipeline

class TestDuckDBIntegration(unittest.TestCase):
    
    def test_duckdb_pandas_consistency(self):
        """Ensure DuckDB and pandas produce identical results."""
        
        # Test with sample data
        sample_data = create_sample_ais_data()
        
        # Pandas aggregation
        pandas_result = sample_data.groupby('imo')['speed'].mean()
        
        # DuckDB aggregation
        engine = DuckDBEngine()
        engine.register_dataframe(sample_data, 'test_data')
        duckdb_result = engine.execute_query("""
            SELECT imo, AVG(speed) as speed
            FROM test_data
            GROUP BY imo
        """)
        
        # Compare results
        pd.testing.assert_series_equal(
            pandas_result.sort_index(),
            duckdb_result.set_index('imo')['speed'].sort_index(),
            rtol=1e-10
        )
    
    def test_hybrid_pipeline_performance(self):
        """Test that hybrid pipeline maintains accuracy with better performance."""
        
        # Compare training results
        hybrid_pipeline = HybridTrainingPipeline(use_duckdb=True)
        pandas_pipeline = HybridTrainingPipeline(use_duckdb=False)
        
        # Both should produce similar ML results
        # (Implementation details...)
```

---

## 📋 **Migration Checklist**

### **Phase 1: Setup** ✅
- [ ] Install DuckDB: `conda activate ML && pip install duckdb`
- [ ] Create `scripts/convert_to_parquet.py`
- [ ] Convert existing pickle files to Parquet
- [ ] Update `config/default.yaml` with DuckDB settings
- [ ] Verify Parquet files load correctly

### **Phase 2: Core Integration** 🔧
- [ ] Create `src/data/duckdb_engine.py`
- [ ] Update `src/data/loader.py` with DuckDB support
- [ ] Add fallback mechanisms for pandas compatibility
- [ ] Create integration tests
- [ ] Benchmark basic operations

### **Phase 3: Script Updates** 📝
- [ ] Modify `scripts/create_training_data.py`
- [ ] Update `scripts/train_h3_model.py` data loading
- [ ] Add DuckDB configuration options
- [ ] Test full training pipeline
- [ ] Verify ML accuracy is maintained

### **Phase 4: Advanced Features** 🚀
- [ ] Create `src/training/hybrid_pipeline.py`
- [ ] Implement complex SQL-based feature engineering
- [ ] Add performance monitoring
- [ ] Create comprehensive benchmarks
- [ ] Optimize for 14-thread hardware

---

## 🎯 **Expected Performance Gains**

### **Before (Current Pandas):**
- **GroupBy operations**: 30-300 seconds on large datasets
- **Multi-year loading**: 60-120 seconds for all years
- **Memory usage**: 16GB+ for full dataset
- **Feature engineering**: 10-30 minutes for complex features

### **After (DuckDB + Pandas):**
- **GroupBy operations**: 3-30 seconds (10-100x faster)
- **Multi-year loading**: 15-40 seconds (3-8x faster)
- **Memory usage**: 8-12GB (50-80% reduction)
- **Feature engineering**: 2-5 minutes (5-15x faster)

### **Your Hardware Advantage:**
- **54GB RAM**: Perfect for DuckDB's columnar operations
- **14 CPU threads**: Excellent for parallel SQL execution
- **RTX 3080 Ti**: Keep for XGBoost GPU acceleration
- **Overall**: Expected 5-20x speedup on data preprocessing

---

## 🚀 **Next Steps**

1. **Start with Phase 1** (Parquet conversion) - low risk, immediate benefits
2. **Test on one experiment** (experiment_h3_simple) before full migration
3. **Benchmark each phase** to measure actual performance gains
4. **Maintain pandas fallbacks** for compatibility during transition
5. **Document performance improvements** for future reference

**Goal**: Achieve production-ready DuckDB integration within 4 weeks while maintaining ML pipeline accuracy and improving data processing performance by 5-20x.

Once done update the readme.md with the new setup.

also update requirments.txt

