"""
Training Data Creation Module

This module contains logic for creating training datasets for vessel trajectory prediction.
Extracted from the inline logic that was previously in scripts and notebooks.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Import our existing modules
sys.path.append(str(Path(__file__).parent.parent))
from features.vessel_h3_tracker import VesselH3Tracker
from features.vessel_features import VesselFeatureExtractor


class SimpleDataCreator:
    """
    Creates simple training data for H3 cell prediction using a single vessel.
    This logic was previously in scripts/create_simple_training_data.py
    """
    
    def __init__(self, h3_resolution: int = 5, data_dir: str = None):
        """
        Initialize the simple data creator.
        
        Args:
            h3_resolution: H3 resolution level for spatial features
            data_dir: Directory containing raw data files
        """
        self.h3_resolution = h3_resolution
        self.data_dir = data_dir or '/home/marius/repo_linux/ais-forecasting/data'
        self.tracker = VesselH3Tracker(h3_resolution=h3_resolution)
        self.extractor = VesselFeatureExtractor(h3_resolution=h3_resolution)
    
    def load_raw_data(self, year: int = 2024) -> pd.DataFrame:
        """Load raw AIS data for the specified year."""
        data_path = f'{self.data_dir}/raw/ais_cape_data_{year}.pkl'
        logging.info(f"Loading raw data from {data_path}")
        return pd.read_pickle(data_path)
    
    def select_test_vessel(self, df: pd.DataFrame, max_records: int = 200) -> Tuple[str, pd.DataFrame]:
        """
        Select a vessel with good data coverage for testing.
        
        Args:
            df: Raw AIS DataFrame
            max_records: Maximum number of records to use
            
        Returns:
            Tuple of (vessel_imo, vessel_data)
        """
        vessel_counts = df['imo'].value_counts()
        test_vessel = vessel_counts.index[0]  # Vessel with most records
        vessel_data = df[df['imo'] == test_vessel].head(max_records).copy()
        
        logging.info(f"Selected vessel {test_vessel} with {len(vessel_data)} records")
        return test_vessel, vessel_data
    
    def create_h3_sequence(self, vessel_data: pd.DataFrame) -> pd.DataFrame:
        """Convert vessel data to H3 sequence."""
        logging.info("Converting to H3 sequence...")
        h3_sequence = self.tracker.convert_vessel_to_h3_sequence(vessel_data)
        logging.info(f"H3 sequence created: {len(h3_sequence)} positions")
        return h3_sequence
    
    def extract_features(self, h3_sequence: pd.DataFrame) -> pd.DataFrame:
        """Extract vessel features from H3 sequence."""
        logging.info("Extracting vessel features...")
        features_df = self.extractor.extract_all_features(h3_sequence)
        logging.info(f"Features extracted: {len(features_df)} rows, {len(features_df.columns)} features")
        return features_df
    
    def create_training_sequences(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create input-target pairs for training.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            DataFrame with training sequences
        """
        logging.info("Creating training sequences...")
        
        sequences = []
        for i in range(len(features_df) - 1):  # -1 because we need next cell
            current_row = features_df.iloc[i]
            next_row = features_df.iloc[i + 1]
            
            # Input features (simplified for baseline)
            input_features = {
                'current_h3_cell': current_row['current_h3_cell'],
                'current_speed': current_row['current_speed'],
                'current_heading': current_row['current_heading'],
                'lat': current_row['lat'],
                'lon': current_row['lon'],
                'time_in_current_cell': current_row['time_in_current_cell']
            }
            
            # Target: next H3 cell
            target = next_row['current_h3_cell']
            
            # Combine
            sequence = {**input_features, 'target_h3_cell': target}
            sequences.append(sequence)
        
        training_df = pd.DataFrame(sequences)
        logging.info(f"Training sequences created: {len(training_df)} samples")
        
        return training_df
    
    def create_simple_training_data(self, output_path: str = None) -> pd.DataFrame:
        """
        Create the complete simple training dataset.
        
        Args:
            output_path: Path to save the training data
            
        Returns:
            DataFrame with training data
        """
        logging.info("🚀 Creating Simple Training Data for H3 Cell Prediction...")
        
        # Load data and select vessel
        df = self.load_raw_data()
        test_vessel, vessel_data = self.select_test_vessel(df)
        
        # Process through pipeline
        h3_sequence = self.create_h3_sequence(vessel_data)
        features_df = self.extract_features(h3_sequence)
        training_df = self.create_training_sequences(features_df)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            training_df.to_pickle(output_path)
            logging.info(f"Training data saved to: {output_path}")
        
        # Print summary
        self._print_summary(training_df, test_vessel)
        
        return training_df
    
    def _print_summary(self, training_df: pd.DataFrame, test_vessel: str):
        """Print summary statistics."""
        unique_cells = training_df['current_h3_cell'].nunique()
        unique_targets = training_df['target_h3_cell'].nunique()
        
        logging.info(f"\n📊 Data Analysis:")
        logging.info(f"   - Test vessel: {test_vessel}")
        logging.info(f"   - Training samples: {len(training_df)}")
        logging.info(f"   - Unique current cells: {unique_cells}")
        logging.info(f"   - Unique target cells: {unique_targets}")
        logging.info(f"   - Average speed: {training_df['current_speed'].mean():.1f} knots")


class MultiVesselDataCreator:
    """
    Creates comprehensive training data using multiple vessels.
    This logic will be used in scripts/create_multi_vessel_training_data.py
    """
    
    def __init__(self, h3_resolution: int = 5, data_dir: str = None):
        """
        Initialize the multi-vessel data creator.
        
        Args:
            h3_resolution: H3 resolution level for spatial features
            data_dir: Directory containing raw data files
        """
        self.h3_resolution = h3_resolution
        self.data_dir = data_dir or '/home/marius/repo_linux/ais-forecasting/data'
        self.tracker = VesselH3Tracker(h3_resolution=h3_resolution)
        self.extractor = VesselFeatureExtractor(h3_resolution=h3_resolution)
    
    def load_all_raw_data(self, years: List[int] = None) -> pd.DataFrame:
        """
        Load raw AIS data for multiple years.
        
        Args:
            years: List of years to load (default: [2024])
            
        Returns:
            Combined DataFrame with all data
        """
        if years is None:
            years = [2024]
        
        all_data = []
        for year in years:
            data_path = f'{self.data_dir}/raw/ais_cape_data_{year}.pkl'
            logging.info(f"Loading data for {year}...")
            df = pd.read_pickle(data_path)
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logging.info(f"Combined dataset: {len(combined_df):,} records from {len(years)} years")
        
        return combined_df
    
    def sample_vessels(self, df: pd.DataFrame, max_vessels: int = None, 
                      max_records_per_vessel: int = 500) -> pd.DataFrame:
        """
        Sample vessels and their data for training.
        
        Args:
            df: Raw AIS DataFrame
            max_vessels: Maximum number of vessels to include
            max_records_per_vessel: Maximum records per vessel
            
        Returns:
            Sampled DataFrame
        """
        vessel_counts = df['imo'].value_counts()
        
        if max_vessels:
            vessels_to_use = vessel_counts.head(max_vessels).index
        else:
            vessels_to_use = vessel_counts.index
        
        logging.info(f"Processing {len(vessels_to_use)} vessels...")
        
        sampled_data = []
        for vessel_imo in vessels_to_use:
            vessel_data = df[df['imo'] == vessel_imo]
            
            # Sample records if needed
            if len(vessel_data) > max_records_per_vessel:
                step = len(vessel_data) // max_records_per_vessel
                vessel_sample = vessel_data.iloc[::step][:max_records_per_vessel]
            else:
                vessel_sample = vessel_data
            
            sampled_data.append(vessel_sample)
        
        combined_sample = pd.concat(sampled_data, ignore_index=True)
        combined_sample = combined_sample.sort_values(['imo', 'mdt']).reset_index(drop=True)
        
        logging.info(f"Sampled dataset: {len(combined_sample):,} records from {len(vessels_to_use)} vessels")
        return combined_sample
    
    def process_all_vessels(self, vessel_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process all vessels through the H3 and feature engineering pipeline.
        
        Args:
            vessel_data: DataFrame with vessel data
            
        Returns:
            Combined DataFrame with all vessel sequences and features
        """
        all_sequences = []
        all_features = []
        
        unique_vessels = vessel_data['imo'].unique()
        logging.info(f"Processing {len(unique_vessels)} vessels through pipeline...")
        
        for i, vessel_imo in enumerate(unique_vessels):
            try:
                # Get vessel data
                single_vessel_data = vessel_data[vessel_data['imo'] == vessel_imo]
                
                # Convert to H3 sequence
                h3_sequence = self.tracker.convert_vessel_to_h3_sequence(single_vessel_data)
                h3_sequence['vessel_imo'] = vessel_imo
                
                # Extract features
                features_df = self.extractor.extract_all_features(h3_sequence)
                features_df['vessel_imo'] = vessel_imo
                
                all_sequences.append(h3_sequence)
                all_features.append(features_df)
                
                if (i + 1) % 10 == 0:
                    logging.info(f"   Processed {i + 1}/{len(unique_vessels)} vessels")
                    
            except Exception as e:
                logging.warning(f"   Failed to process vessel {vessel_imo}: {e}")
                continue
        
        # Combine results
        combined_sequences = pd.concat(all_sequences, ignore_index=True)
        combined_features = pd.concat(all_features, ignore_index=True)
        
        logging.info(f"Pipeline complete: {len(combined_features)} feature records from {len(all_features)} vessels")
        
        return combined_sequences, combined_features
    
    def create_multi_vessel_training_data(self, years: List[int] = None, 
                                        max_vessels: int = 50,
                                        output_path: str = None) -> pd.DataFrame:
        """
        Create comprehensive multi-vessel training dataset.
        
        Args:
            years: Years to include in training data
            max_vessels: Maximum number of vessels to process
            output_path: Path to save the training data
            
        Returns:
            DataFrame with multi-vessel training data
        """
        logging.info("🚀 Creating Multi-Vessel Training Data...")
        
        # Load and sample data
        df = self.load_all_raw_data(years)
        sampled_data = self.sample_vessels(df, max_vessels=max_vessels)
        
        # Process through pipeline
        sequences, features = self.process_all_vessels(sampled_data)
        
        # Create training sequences (simplified approach)
        training_sequences = []
        for vessel_imo in features['vessel_imo'].unique():
            vessel_features = features[features['vessel_imo'] == vessel_imo].sort_values('mdt')
            
            for i in range(len(vessel_features) - 1):
                current = vessel_features.iloc[i]
                next_row = vessel_features.iloc[i + 1]
                
                sequence = {
                    'vessel_imo': vessel_imo,
                    'current_h3_cell': current['current_h3_cell'],
                    'current_speed': current['current_speed'],
                    'current_heading': current['current_heading'],
                    'lat': current['lat'],
                    'lon': current['lon'],
                    'time_in_current_cell': current['time_in_current_cell'],
                    'target_h3_cell': next_row['current_h3_cell']
                }
                training_sequences.append(sequence)
        
        training_df = pd.DataFrame(training_sequences)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            training_df.to_pickle(output_path)
            logging.info(f"Multi-vessel training data saved to: {output_path}")
        
        # Print summary
        self._print_multi_vessel_summary(training_df)
        
        return training_df
    
    def _print_multi_vessel_summary(self, training_df: pd.DataFrame):
        """Print summary statistics for multi-vessel data."""
        unique_vessels = training_df['vessel_imo'].nunique()
        unique_cells = training_df['current_h3_cell'].nunique()
        unique_targets = training_df['target_h3_cell'].nunique()
        
        logging.info(f"\n📊 Multi-Vessel Data Analysis:")
        logging.info(f"   - Vessels: {unique_vessels}")
        logging.info(f"   - Training samples: {len(training_df):,}")
        logging.info(f"   - Unique current cells: {unique_cells}")
        logging.info(f"   - Unique target cells: {unique_targets}")
        logging.info(f"   - Average speed: {training_df['current_speed'].mean():.1f} knots")
        logging.info(f"   - Samples per vessel: {len(training_df) / unique_vessels:.1f}")
