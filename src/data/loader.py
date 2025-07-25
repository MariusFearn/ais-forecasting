import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import glob
from pathlib import Path
import pickle
from pytorch_forecasting import TimeSeriesDataSet


class AISDataLoader:
    """
    Utility class for loading and processing AIS data.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader with the data directory.
        
        Args:
            data_dir: Directory containing raw and processed data
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def _find_files(self, pattern: str, directory: str) -> List[str]:
        """
        Find files matching the given pattern in the specified directory.
        
        Args:
            pattern: Glob pattern for file matching
            directory: Directory to search in
            
        Returns:
            List[str]: List of matching file paths
        """
        search_pattern = os.path.join(directory, pattern)
        return glob.glob(search_pattern)
    
    def list_raw_files(self, pattern: str = "*.csv") -> List[str]:
        """
        List raw data files matching the given pattern.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List[str]: List of matching file paths
        """
        return self._find_files(pattern, self.raw_dir)
    
    def list_processed_files(self, pattern: str = "*.pkl") -> List[str]:
        """
        List processed data files matching the given pattern.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List[str]: List of matching file paths
        """
        return self._find_files(pattern, self.processed_dir)
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw data from the specified file.
        
        Args:
            file_path: Path to the raw data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        # Determine file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return df
    
    def load_processed_data(self, file_name: str) -> pd.DataFrame:
        """
        Load processed data from the processed directory.
        
        Args:
            file_name: Name of the processed data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        # Ensure file_name has the correct extension
        if not file_name.endswith(".pkl"):
            file_name = f"{file_name}.pkl"
        
        file_path = os.path.join(self.processed_dir, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed data file not found: {file_path}")
        
        return pd.read_pickle(file_path)
    
    def save_processed_data(self, df: pd.DataFrame, file_name: str):
        """
        Save processed data to the processed directory.
        
        Args:
            df: DataFrame to save
            file_name: Name of the output file
        """
        # Ensure file_name has the correct extension
        if not file_name.endswith(".pkl"):
            file_name = f"{file_name}.pkl"
        
        file_path = os.path.join(self.processed_dir, file_name)
        
        df.to_pickle(file_path)
        print(f"Saved processed data to: {file_path}")
    
    def create_time_series_dataset(self, 
                                  df: pd.DataFrame,
                                  time_idx: str,
                                  target: str,
                                  group_ids: List[str],
                                  max_encoder_length: int,
                                  max_prediction_length: int,
                                  static_categoricals: List[str] = None,
                                  static_reals: List[str] = None,
                                  time_varying_known_reals: List[str] = None,
                                  time_varying_known_categoricals: List[str] = None,
                                  time_varying_unknown_reals: List[str] = None,
                                  time_varying_unknown_categoricals: List[str] = None,
                                  add_relative_time_idx: bool = True,
                                  add_target_scales: bool = True,
                                  add_encoder_length: bool = True) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Create TimeSeriesDataSet for training and validation.
        
        Args:
            df: DataFrame with the time series data
            time_idx: Column name for time index
            target: Column name for target variable
            group_ids: List of column names identifying a time series
            max_encoder_length: Maximum encoder length
            max_prediction_length: Maximum prediction length
            static_categoricals: List of static categorical variables
            static_reals: List of static real variables
            time_varying_known_reals: List of time-varying known real variables
            time_varying_known_categoricals: List of time-varying known categorical variables
            time_varying_unknown_reals: List of time-varying unknown real variables
            time_varying_unknown_categoricals: List of time-varying unknown categorical variables
            add_relative_time_idx: If True, add relative time index
            add_target_scales: If True, add target scales
            add_encoder_length: If True, add encoder length
            
        Returns:
            Tuple[TimeSeriesDataSet, TimeSeriesDataSet]: Training and validation datasets
        """
        # Define the training cutoff (e.g., use the last 30 days for validation)
        validation_cutoff = df[time_idx].max() - max_prediction_length
        
        # Initialize kwargs with default empty lists for None values
        dataset_kwargs = {
            'time_idx': time_idx,
            'target': target,
            'group_ids': group_ids,
            'max_encoder_length': max_encoder_length,
            'max_prediction_length': max_prediction_length,
            'static_categoricals': static_categoricals or [],
            'static_reals': static_reals or [],
            'time_varying_known_reals': time_varying_known_reals or [],
            'time_varying_known_categoricals': time_varying_known_categoricals or [],
            'time_varying_unknown_reals': time_varying_unknown_reals or [],
            'time_varying_unknown_categoricals': time_varying_unknown_categoricals or [],
            'add_relative_time_idx': add_relative_time_idx,
            'add_target_scales': add_target_scales,
            'add_encoder_length': add_encoder_length,
        }
        
        # Create the training dataset
        training = TimeSeriesDataSet(
            df[df[time_idx] <= validation_cutoff],
            **dataset_kwargs
        )
        
        # Create the validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training, df, predict=True, stop_randomization=True
        )
        
        return training, validation
