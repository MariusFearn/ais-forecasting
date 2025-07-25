import os
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, NBeats
from pytorch_forecasting.metrics import SMAPE
from typing import Dict, Any, Tuple, List, Optional

from src.models.base_model import BaseTimeSeriesModel


class NBeatsModel(BaseTimeSeriesModel):
    """
    N-BEATS model implementation.
    Wraps PyTorch Forecasting's NBeats with our project interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the N-BEATS model with the given configuration.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        self.config = config
        self.max_prediction_length = config.get("max_prediction_length", 7)
        self.max_encoder_length = config.get("max_encoder_length", 30)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.loss = config.get("loss", SMAPE())
        self.batch_size = config.get("batch_size", 64)
        self.max_epochs = config.get("max_epochs", 50)
        self.widths = config.get("widths", [32, 512])
        self.backcast_loss_ratio = config.get("backcast_loss_ratio", 0.0)
        
        self.training_dataset = None
        self.model = None
        self.trainer = None
    
    def _create_datasets(self, df_data):
        """
        Create training and validation datasets.
        
        Args:
            df_data: Pandas DataFrame with the input data
        
        Returns:
            tuple: (training_dataset, validation_dataset)
        """
        # Define the cutoff for training/validation split
        cut_days = self.config.get("days_left_for_testing", 30)
        training_cutoff = df_data["time_idx"].max() - cut_days
        
        # Create the training dataset
        training = TimeSeriesDataSet(
            df_data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="value",  # Adjust based on your actual target column
            group_ids=["GroupIDS"],  # Adjust based on your actual group identifiers
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=["month", "day_of_week"],  # Adjust as needed
            time_varying_unknown_reals=["value"],  # Adjust based on your target
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        # Create validation dataset from the training dataset
        validation = TimeSeriesDataSet.from_dataset(
            training, df_data, predict=True, stop_randomization=True
        )
        
        self.training_dataset = training
        return training, validation
    
    def fit(self, train_dataloader=None, val_dataloader=None, df_data=None):
        """
        Train the model with the provided data.
        
        Args:
            train_dataloader: Optional DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            df_data: Optional DataFrame to create datasets if dataloaders not provided
            
        Returns:
            self: Trained model instance
        """
        if train_dataloader is None or val_dataloader is None:
            if df_data is None:
                raise ValueError("Either dataloaders or df_data must be provided")
            
            training, validation = self._create_datasets(df_data)
            
            # Create dataloaders
            train_dataloader = training.to_dataloader(
                train=True, batch_size=self.batch_size, num_workers=self.config.get("num_workers", 4)
            )
            val_dataloader = validation.to_dataloader(
                train=False, batch_size=self.batch_size, num_workers=self.config.get("num_workers", 4)
            )
        
        # Initialize the N-BEATS model
        self.model = NBeats.from_dataset(
            self.training_dataset,
            learning_rate=self.learning_rate,
            log_interval=self.config.get("log_interval", 100),
            log_val_interval=self.config.get("log_val_interval", 1),
            widths=self.widths,
            backcast_loss_ratio=self.backcast_loss_ratio,
            loss=self.loss,
        )
        
        # Set up callbacks
        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=self.config.get("patience", 10), 
                verbose=True, 
                mode="min"
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
        ]
        
        # Set up logger
        logger = pl.loggers.TensorBoardLogger(
            save_dir=self.config.get("log_dir", "tensorboard_logs"),
            name=self.config.get("experiment_name", "nbeats_model"),
        )
        
        # Initialize trainer
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.config.get("accelerator", "auto"),
            enable_model_summary=True,
            gradient_clip_val=self.config.get("gradient_clip_val", 0.1),
            callbacks=callbacks,
            logger=logger,
        )
        
        # Train the model
        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        return self
    
    def predict(self, dataloader) -> torch.Tensor:
        """
        Generate forecasts for the provided data.
        
        Args:
            dataloader: DataLoader with the input data
            
        Returns:
            torch.Tensor: Model predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(dataloader)
        return predictions
    
    def save(self, path: str):
        """
        Save the model to the specified path.
        
        Args:
            path: Directory where the model should be saved
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        torch.save(self.model.state_dict(), path)
    
    @classmethod
    def load(cls, path: str, config: Dict[str, Any], training_dataset=None):
        """
        Load a trained model from the specified path.
        
        Args:
            path: Path to the saved model
            config: Model configuration
            training_dataset: Training dataset used to create the model
            
        Returns:
            NBeatsModel: Loaded model instance
        """
        instance = cls(config)
        
        if training_dataset is None:
            raise ValueError("training_dataset must be provided to load an N-BEATS model")
        
        instance.training_dataset = training_dataset
        
        # Initialize the model from the dataset
        instance.model = NBeats.from_dataset(
            training_dataset,
            learning_rate=instance.learning_rate,
            log_interval=instance.config.get("log_interval", 100),
            log_val_interval=instance.config.get("log_val_interval", 1),
            widths=instance.widths,
            backcast_loss_ratio=instance.backcast_loss_ratio,
            loss=instance.loss,
        )
        
        # Load the saved weights
        state_dict = torch.load(path)
        instance.model.load_state_dict(state_dict)
        
        return instance
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluate the model on the provided data and return metrics.
        
        Args:
            dataloader: DataLoader with the test data
            
        Returns:
            Dict[str, float]: Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Run evaluation
        predictions = self.predict(dataloader)
        
        # Calculate metrics
        actuals = torch.cat([y for x, (y, weight) in iter(dataloader)])
        metrics = {
            "mae": ((predictions - actuals).abs().mean()).item(),
            "rmse": ((predictions - actuals).pow(2).mean().sqrt()).item(),
        }
        
        return metrics
