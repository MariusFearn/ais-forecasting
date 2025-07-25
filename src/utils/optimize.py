from typing import Dict, Any, Callable, List, Optional, Union
import optuna
import logging
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import os


class HyperparameterOptimization:
    """
    Class for hyperparameter optimization using Optuna.
    """
    
    def __init__(self, 
                 trial_function: Callable[[optuna.Trial], float], 
                 param_space: Dict[str, Any],
                 n_trials: int = 50,
                 study_name: str = "hyperparameter_optimization",
                 direction: str = "minimize",
                 storage: Optional[str] = None,
                 load_if_exists: bool = True):
        """
        Initialize the hyperparameter optimization.
        
        Args:
            trial_function: Function that takes an Optuna trial and returns a score
            param_space: Dictionary defining the parameter space
            n_trials: Number of trials to run
            study_name: Name of the optimization study
            direction: Direction of optimization (minimize or maximize)
            storage: Database URL for Optuna storage
            load_if_exists: Whether to load existing study or create a new one
        """
        self.trial_function = trial_function
        self.param_space = param_space
        self.n_trials = n_trials
        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        self.load_if_exists = load_if_exists
        
        self.study = None
        
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters for the current trial based on the parameter space.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dict[str, Any]: Dictionary with suggested parameters
        """
        suggested_params = {}
        
        for param_name, param_config in self.param_space.items():
            param_type = param_config["type"]
            
            if param_type == "categorical":
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            elif param_type == "int":
                suggested_params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"], step=param_config.get("step", 1)
                )
            elif param_type == "float":
                if param_config.get("log", False):
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"], log=True
                    )
                else:
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"], step=param_config.get("step", None)
                    )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
                
        return suggested_params
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Score value to be minimized or maximized
        """
        params = self._suggest_parameters(trial)
        score = self.trial_function(trial, params)
        return score
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the hyperparameter optimization.
        
        Returns:
            Dict[str, Any]: Best parameters found
        """
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
        )
        
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logging.info(f"Best value: {best_value}")
        logging.info(f"Best hyperparameters: {best_params}")
        
        return best_params
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found during optimization.
        
        Returns:
            Dict[str, Any]: Best parameters found
        """
        if self.study is None:
            raise ValueError("Optimization has not been run yet. Call optimize() first.")
        
        return self.study.best_params


def create_optuna_callbacks(trial: optuna.Trial, 
                           model_dir: str, 
                           experiment_name: str) -> List[Any]:
    """
    Create callbacks for PyTorch Lightning training with Optuna integration.
    
    Args:
        trial: Optuna trial object
        model_dir: Directory to save model checkpoints
        experiment_name: Name of the experiment
        
    Returns:
        List[Any]: List of callbacks
    """
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(model_dir, f"trial_{trial.number}"),
        filename=f"{experiment_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    
    logger = TensorBoardLogger(
        save_dir=os.path.join(model_dir, "logs"),
        name=f"{experiment_name}_trial_{trial.number}",
    )
    
    # Early stopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode="min",
    )
    
    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    
    # Optuna pruning integration
    optuna_pruning = optuna.integration.PyTorchLightningPruningCallback(
        trial, monitor="val_loss"
    )
    
    return [checkpoint_callback, early_stop_callback, lr_monitor, optuna_pruning], logger
