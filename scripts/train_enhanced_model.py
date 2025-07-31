#!/usr/bin/env python3
"""
Train enhanced multi-vessel H3 cell prediction model.

This is a thin wrapper around src/training/enhanced_trainer.py
following the src/scripts convention.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.enhanced_trainer import EnhancedModelTrainer


def main():
    """Train enhanced multi-vessel H3 predictor."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize enhanced trainer with optimized config
        enhanced_config = {
            'n_estimators': 100,
            'max_depth': 15,
            'random_state': 42,
            'n_jobs': -1  # Use all available cores
        }
        
        trainer = EnhancedModelTrainer(model_config=enhanced_config)
        
        # Train enhanced model
        training_path = '/home/marius/repo_linux/ais-forecasting/data/processed/training_sets/multi_vessel_h3_sequences.pkl'
        output_dir = '/home/marius/repo_linux/ais-forecasting/data/models/final_models'
        
        model, encoders, metrics = trainer.train_enhanced_h3_predictor(
            training_path=training_path,
            output_dir=output_dir
        )
        
        logging.info("✅ Enhanced model training completed successfully!")
        logging.info(f"🎯 Test Accuracy: {metrics['test_accuracy']:.1%}")
        logging.info(f"🚢 Trained on {metrics['n_vessels']} vessels")
        logging.info(f"🗺️ Predicting among {metrics['n_classes']} H3 cells")
        logging.info("🚀 Enhanced multi-vessel model ready for predictions!")
        
        return model, encoders, metrics
        
    except Exception as e:
        logging.error(f"❌ Enhanced model training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()