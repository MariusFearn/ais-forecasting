#!/usr/bin/env python3
"""
Create multi-vessel training data for H3 cell prediction.

This is a thin wrapper around src/training/data_creator.py
following the src/scripts convention.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.data_creator import MultiVesselDataCreator


def main():
    """Create comprehensive multi-vessel training data."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize data creator
        data_creator = MultiVesselDataCreator(h3_resolution=5)
        
        # Create multi-vessel training data
        output_path = '/home/marius/repo_linux/ais-forecasting/data/processed/training_sets/multi_vessel_h3_sequences.pkl'
        
        training_df = data_creator.create_multi_vessel_training_data(
            years=[2024],  # Start with 2024 data
            max_vessels=50,  # Limit for initial testing
            output_path=output_path
        )
        
        logging.info("✅ Multi-vessel training data creation completed successfully!")
        logging.info(f"🚀 Created {len(training_df):,} training samples from multiple vessels!")
        logging.info("🎯 Ready for enhanced model training!")
        
        return training_df
        
    except Exception as e:
        logging.error(f"❌ Failed to create multi-vessel training data: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()