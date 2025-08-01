"""Hybrid pandas + DuckDB pipeline for optimal performance."""
import pandas as pd
from typing import Dict
from pathlib import Path
from src.data.duckdb_engine import DuckDBEngine
from src.training.pipeline import TrainingPipeline

class HybridTrainingPipeline:
    """Training pipeline that uses DuckDB for data ops, pandas for ML."""
    
    def __init__(self, use_duckdb: bool = True):
        self.use_duckdb = use_duckdb
        if use_duckdb:
            self.duckdb_engine = DuckDBEngine()
        
        # Keep pandas pipeline for ML operations
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
