# Models Directory

This directory contains trained model artifacts and metadata.

## Model Organization

```
models/
├── experiments/
│   ├── experiment_001/
│   │   ├── tft_model.pth
│   │   ├── model_config.yaml
│   │   ├── training_metrics.json
│   │   └── evaluation_results.json
│   └── experiment_002/
│       └── ...
├── production/
│   ├── best_tft_model.pth
│   ├── best_nbeats_model.pth
│   └── model_metadata.json
└── checkpoints/
    ├── tft_epoch_10.ckpt
    └── nbeats_epoch_15.ckpt
```

## Model Artifacts

### Required Files per Model
- `*.pth`: PyTorch model state dict
- `model_config.yaml`: Model configuration parameters
- `training_metrics.json`: Training history and metrics
- `evaluation_results.json`: Validation/test performance

### Model Metadata
- Model architecture details
- Training data version
- Hyperparameters used
- Performance metrics
- Training timestamp
- Model size and inference time

## Naming Convention

- **Experiment models**: `{model_type}_exp_{experiment_id}.pth`
- **Production models**: `{model_type}_prod_v{version}.pth`
- **Checkpoints**: `{model_type}_epoch_{epoch}.ckpt`

## Model Versioning

Use semantic versioning for production models:
- Major: Architecture changes
- Minor: Feature engineering changes
- Patch: Hyperparameter tuning
