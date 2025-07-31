# Scripts module initialization

import pandas as pd

# Try to import pytorch forecasting components with graceful fallback
try:
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    PYTORCH_FORECASTING_AVAILABLE = True
except ImportError:
    PYTORCH_FORECASTING_AVAILABLE = False
    print("⚠️  pytorch_forecasting not available. Some advanced models will not work.")
    print("   Install with: pip install pytorch-forecasting")

# This file contains import configurations for the scripts module
# The actual training scripts are in individual .py files in this directory