import unittest
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add src to path for local imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.loader import AISDataLoader
from data.duckdb_engine import DuckDBEngine

class TestDuckDBIntegration(unittest.TestCase):
    """
    Integration tests for the DuckDB engine and its integration with the AISDataLoader.
    
    These tests verify that the DuckDB-powered data loading and processing pipeline
    produces correct and consistent results compared to the baseline pandas operations.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        print("\nSetting up DuckDB integration test environment...")
        
        # --- Load Configuration ---
        # We use a simple config to ensure the test runs quickly.
        with open("config/experiment_configs/experiment_h3_simple.yaml", 'r') as f:
            cls.config = yaml.safe_load(f)
            
        # Load default config to get base paths and other settings
        with open("config/default.yaml", 'r') as f:
            default_config = yaml.safe_load(f)
            # Deep merge config
            cls.config = {**default_config, **cls.config}

        # --- Initialize Loaders ---
        cls.data_dir = cls.config['data']['data_dir']
        
        # DuckDB-powered loader (the system under test)
        cls.duckdb_loader = AISDataLoader(data_dir=cls.data_dir, use_duckdb=True)
        
        # Pure pandas loader (for baseline comparison)
        cls.pandas_loader = AISDataLoader(data_dir=cls.data_dir, use_duckdb=False)

        # --- Load Test Data ---
        # We'll use a single year of data for these tests.
        cls.test_year = '2023'
        
        print(f"Loading test data for year {cls.test_year}...")
        # Load data using both methods to have them ready for comparison
        cls.duckdb_df = cls.duckdb_loader.load_multi_year_data_optimized([cls.test_year])
        
        # The original loader uses pickle, so we load the parquet file with pandas for a fair comparison
        parquet_path = Path(cls.data_dir) / "raw" / "parquet" / f"ais_cape_data_{cls.test_year}.parquet"
        cls.pandas_df = pd.read_parquet(parquet_path)
        
        print("Test setup complete.")

    def test_data_loading_consistency(self):
        """
        Verify that DuckDB and pandas loaders return DataFrames with the same shape and columns.
        """
        print("\nTesting data loading consistency...")
        
        # Check that both dataframes are loaded
        self.assertIsNotNone(self.duckdb_df, "DuckDB DataFrame should not be None.")
        self.assertIsNotNone(self.pandas_df, "Pandas DataFrame should not be None.")
        
        # Check for the same number of columns
        self.assertEqual(self.duckdb_df.shape[1], self.pandas_df.shape[1],
                         "DataFrames should have the same number of columns.")
                         
        # Check for the same column names
        self.assertListEqual(sorted(self.duckdb_df.columns.tolist()), sorted(self.pandas_df.columns.tolist()),
                             "DataFrames should have the same column names.")
        
        print("   ✅ Passed: Loading consistency.")

    def test_duckdb_engine_aggregation(self):
        """
        Test a simple aggregation (COUNT and AVG) with DuckDB and compare to pandas.
        """
        print("\nTesting DuckDB engine aggregation...")
        
        # --- DuckDB Aggregation ---
        # Use the DuckDB engine directly to perform an aggregation
        duckdb_engine = self.duckdb_loader.duckdb_engine
        query = f"""
        SELECT
            imo,
            COUNT(imo) as record_count,
            AVG(speed) as avg_speed
        FROM '{self.duckdb_engine.db_path}'
        WHERE imo IS NOT NULL
        GROUP BY imo
        ORDER BY record_count DESC
        LIMIT 10
        """
        duckdb_result = duckdb_engine.query_to_df(query)
        
        # --- Pandas Equivalent Aggregation ---
        pandas_agg = self.pandas_df.groupby('imo').agg(
            record_count=('imo', 'count'),
            avg_speed=('speed', 'mean')
        ).sort_values('record_count', ascending=False).head(10).reset_index()

        # --- Comparison ---
        self.assertIsNotNone(duckdb_result, "DuckDB aggregation result should not be None.")
        self.assertEqual(len(duckdb_result), 10, "DuckDB aggregation should return 10 rows.")
        
        # Compare the list of top vessels (IMOs)
        self.assertListEqual(duckdb_result['imo'].tolist(), pandas_agg['imo'].tolist(),
                             "Top 10 vessels (IMOs) should be identical.")
                             
        # Compare the calculated values (record count and average speed)
        # We use `np.testing.assert_allclose` for robust floating-point comparison
        pd.testing.assert_frame_equal(
            duckdb_result.set_index('imo'),
            pandas_agg.set_index('imo'),
            check_exact=False,
            atol=1e-5 # Absolute tolerance for floating point comparison
        )
        
        print("   ✅ Passed: Aggregation correctness.")

    def test_filtered_loading(self):
        """
        Test that loading data with a filter works correctly.
        """
        print("\nTesting filtered data loading...")
        
        # Define a filter to select a single, specific vessel
        target_imo = self.pandas_df['imo'].value_counts().idxmax() # Get the most frequent IMO
        filters = [f"imo = '{target_imo}'"]
        
        # Load filtered data using the DuckDB loader
        filtered_df_duckdb = self.duckdb_loader.load_multi_year_data_optimized(
            [self.test_year], filters=filters
        )
        
        # Perform the same filtering with pandas
        filtered_df_pandas = self.pandas_df[self.pandas_df['imo'] == target_imo]
        
        # --- Comparison ---
        self.assertGreater(len(filtered_df_duckdb), 0, "Filtered DuckDB load should return data.")
        
        # Verify that the number of records matches exactly
        self.assertEqual(len(filtered_df_duckdb), len(filtered_df_pandas),
                         "Filtered record counts should be identical.")
                         
        # Verify that all records in the filtered set belong to the target vessel
        self.assertTrue((filtered_df_duckdb['imo'] == target_imo).all(),
                        "All records in the filtered DataFrame should belong to the target IMO.")

        print("   ✅ Passed: Filtered loading.")

if __name__ == '__main__':
    unittest.main()
