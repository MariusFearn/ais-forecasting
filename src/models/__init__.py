from .base_model import BaseTimeSeriesModel

# Only import if dependencies are available
try:
    from .tft_model import TFTModel
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

try:
    from .nbeats_model import NBeatsModel
    NBEATS_AVAILABLE = True
except ImportError:
    NBEATS_AVAILABLE = False

# Dynamic __all__ based on available imports
__all__ = ['BaseTimeSeriesModel']
if TFT_AVAILABLE:
    __all__.append('TFTModel')
if NBEATS_AVAILABLE:
    __all__.append('NBeatsModel')
