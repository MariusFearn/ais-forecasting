# AIS Vessel Trajectory Prediction - Quick Reference

## 🎯 Development Workflow (The Golden Path)
When adding a new feature, the standard process is:
1.  **Add Core Logic:** Place new reusable functions and classes in the appropriate module within the `src/` directory.
2.  **Create or Update Config:** If the feature requires new parameters, add them to `config/default.yaml` or create a new experiment-specific YAML file in `config/experiment_configs/`.
3.  **Update Scripts:** Modify existing scripts in `scripts/` to use the new logic and configurations. Create new scripts only when introducing a completely new, high-level task.
4.  **Add Tests:** Create corresponding unit or integration tests in the `tests/` directory to validate the new feature.
5.  **Update Documentation:** Document the new feature in the main `README.md` and any other relevant files.

## ✍️ Coding Style and Conventions
- **File Paths:** Always use `pathlib.Path` for handling file and directory paths to ensure cross-platform compatibility.
- **Logging:** Use the `logging` module for all output in scripts and source code. Avoid using `print()` for logging purposes.
- **Code Formatting:** Adhere to PEP 8 standards for all Python code to maintain consistency and readability.
- **Type Hinting:** Use type hints for all function signatures to improve code clarity and allow for static analysis.

## 📦 Dependency Management
- **Pinning Dependencies:** When adding or updating a library in `requirements.txt`, pin it to the exact version (e.g., `pandas==2.3.0`). This ensures that the environment is 100% reproducible.
- **Updating `requirements.txt`:** Use a tool like `pip-tools` to manage dependencies. After installing a new package (`pip install new-package`), run `pip freeze > requirements.txt` to capture the exact versions of all packages in the environment.

---

## 📁 Project sturckture - we do not change this strukture - keep attention if you create new files to put them in the right place
ais-forecasting/
├── .github/                    # Contains GitHub-specific files, like CI/CD workflows.
│
├── config/                     # Stores all project configuration files.
│   ├── default.yaml            # Default parameters for the entire project.
│   └── experiment_configs/     # Configurations for specific machine learning experiments.
│       ├── experiment_nbeats.yaml # Settings for an N-BEATS model experiment.
│       └── experiment_tft.yaml    # Settings for a Temporal Fusion Transformer experiment.
│
├── data/                       # Holds all data used in the project.
│   ├── raw/                    # Raw, immutable data. Original data files (.pkl) are stored here.
│   │   └── parquet/            # Parquet versions of raw data for optimized access.
│   ├── processed/              # Cleaned, transformed, and feature-engineered data.
│   │   ├── training_sets/      # Final datasets ready for model training.
│   │   ├── vessel_features/    # Intermediate features extracted for each vessel.
│   │   └── predictions/        # Stores the output predictions from models.
│   └── models/                 # Contains all trained model artifacts.
│       ├── final_models/       # Serialized, production-ready models.
│       ├── checkpoints/        # Saved states during large model training.
│       └── hyperparameter_logs/# Logs from hyperparameter optimization runs.
│
├── experiments/                # Tracks results and artifacts from ML experiments. (holds variables and setting for experiments)
│   ├── baseline_experiments/   # Results from simple baseline models.
│   ├── nbeats_experiments/     # Results from N-BEATS model experiments.
│   └── tft_experiments/        # Results from TFT model experiments.
│
├── notebooks/                  # Jupyter notebooks for interactive analysis and visualization. (data analysis, research, presentations)
│   ├── exploratory.ipynb       # Initial data exploration and analysis.
│   ├── preprocessing.ipynb     # Interactive data cleaning and preparation.
│   ├── model_development.ipynb # Prototyping and developing new models.
│   ├── evaluation.ipynb        # In-depth evaluation of model performance.
│   └── visual_training_analysis.ipynb # Visualizing the full training pipeline.
│
├── scripts/                    # Contains standalone, executable scripts for core tasks. (should not containe functions, use src for functions - should try not have code duplication)
│   ├── create_simple_training_data.py # Generates a small, single-vessel dataset.
│   ├── train_simple_model.py   # Trains a baseline model on the simple dataset.
│   ├── create_multi_vessel_training_data.py # Generates the full training dataset.
│   ├── train_enhanced_model.py # Trains the primary, enhanced model.
│   ├── evaluate.py             # Runs model evaluation from the command line.
│   ├── predict.py              # Runs predictions using a trained model.
│   ├── test_simple.py          # A simple test script for quick validation.
│   ├── create_training_data.py # Delete: Redundant, functionality is split.
│   ├── train.py                # Delete: Redundant, functionality is split.
│   └── quick_start_h3.py       # Delete: Old script, functionality now in notebooks.
│
├── src/                        # Contains all the project's source code as a Python package. (all code and functions goes here - script gets data and runs functions from src)
│   ├── __init__.py             # Makes 'src' a package, allowing imports.
│   ├── data/                   # Modules for data loading and preprocessing.
│   ├── features/               # Modules for feature engineering and transformation.
│   ├── models/                 # Python definitions of model architectures.
│   ├── utils/                  # Reusable utility functions and helper classes.
│   └── visualization/          # Code for generating plots and maps.
│
├── tests/                      # Contains all tests for the project source code.
│   ├── test_data.py            # Unit tests for data loading and validation.
│   ├── test_features.py        # Unit tests for the feature engineering pipeline.
│   └── test_models.py          # Unit tests for model input/output validation.
│
├── visualizations/             # Stores saved output plots, maps, and other visuals.
│   ├── *.html                  # Interactive maps and plots generated by notebooks/scripts.
│   └── ultra_fast_maritime_visualization.py # Move: This is a script, not a visualization.
│
├── README.md                   # This file: The main documentation for the project.
├── requirements.txt            # A list of all Python packages required to run the project. Update this if we need new models and make sure we are in ML environment if we install something.
└── .gitignore                  # Specifies files and folders to be ignored by Git.


In terminal we use conda ML - not base - conda activate ML must be run before running any scripts or notebooks.
