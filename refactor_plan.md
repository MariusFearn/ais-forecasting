# Refactoring Plan: Configuration System Overhaul

## 1. Objective

This plan outlines the steps to refactor the YAML configuration system and associated Python scripts. The goals are to:
- **Increase Consistency:** Standardize how configurations are defined.
- **Reduce Duplication:** Consolidate common parameters into base configurations.
- **Improve Maintainability:** Make the system easier to understand, manage, and extend.

## 2. Refactoring Steps

### Step 2.1: Consolidate Training Configurations into a Base File

- **Action:** Create a new file `config/experiment_configs/base_h3_experiment.yaml` to hold parameters common to all H3 prediction training runs.
- **Rationale:** The `simple`, `comprehensive`, and `massive` training configurations share many parameters. Centralizing them reduces redundancy and makes experiment-specific files cleaner.

**`base_h3_experiment.yaml` will contain:**
```yaml
# Base configuration for all H3 prediction experiments
defaults:
  - ../default

data:
  target: "target_h3_cell"
  exclude_columns: ["target_h3_cell", "vessel_imo", "data_year"]
  random_state: 42

training:
  apply_phase5_fixes: true
  handle_single_sample_classes: true

feature_selection:
  method: "mutual_info_classif"
  
output:
  model_path_template: "data/models/final_models/{experiment_name}_predictor.pkl"
  encoder_path_template: "data/models/final_models/{experiment_name}_encoder.pkl"
  metadata_path_template: "data/models/final_models/{experiment_name}_metadata.pkl"

evaluation:
  include_feature_importance: true
  include_distance_evaluation: true
  sample_prediction_test: true
```

- **Action:** Update `simple_h3_experiment.yaml`, `comprehensive_h3_experiment.yaml`, and `massive_h3_experiment.yaml` to inherit from this new base file.

### Step 2.2: Clarify and Rename Default Configuration

- **Action:** Rename `config/default.yaml` to `config/dl_default.yaml`.
- **Rationale:** The parameters in `default.yaml` are specific to the PyTorch Forecasting (deep learning) models. Renaming it clarifies its purpose and prevents confusion with the tree-based models.
- **Action:** Update `nbeats_experiment.yaml` and `tft_experiment.yaml` to inherit from `dl_default.yaml`.

### Step 2.3: Standardize Path Definitions

- **Action:** Add a new `paths` section to `config/default.yaml` (which will be the new general default, not the DL-specific one). This will centralize the main directory paths.
- **Rationale:** Avoids hardcoding directory structures in multiple files, making the project more robust if the structure changes.

**New `config/default.yaml`:**
```yaml
paths:
  models: "data/models/final_models"
  processed_data: "data/processed/training_sets"
  raw_data: "data/raw"
  experiments: "experiments"
  evaluation_results: "experiments/evaluation_results"
```

### Step 2.4: Update Python Scripts

- **Action:** Modify `scripts/train_h3_model.py` to:
    - Correctly load configurations that inherit from the new base.
    - Construct output paths using the new `paths` variables and the `model_path_template` from the config.
- **Action:** Modify `scripts/evaluate_model.py` to:
    - Use the new `paths` variables to locate models and data.
- **Action:** Modify `scripts/create_training_data.py` to:
    - Use the new `paths` variables for input and output directories.

## 3. Verification Plan

After the refactoring is complete, the following steps will be taken to ensure the system still functions correctly:
1.  **Run Data Creation:** Execute `create_training_data.py` for the `simple` and `comprehensive` configurations.
2.  **Run Training:** Execute `train_h3_model.py` for the `simple` and `comprehensive` configurations.
3.  **Run Testing:** Execute `test_system.py` for `infrastructure` and `model_performance` tests.
4.  **Run Evaluation:** Execute `evaluate_model.py` for `simple` and `comprehensive` evaluations.

All steps will be checked to ensure they complete without errors and produce the expected artifacts in the correct locations.
