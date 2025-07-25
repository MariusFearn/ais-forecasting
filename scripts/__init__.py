# filepath: /workspaces/Deep-learning-models/main_model/2_1_build_model_tft.py

import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Load and prepare the data
df_data = pd.read_pickle(r'./data/hexbingdata.pkl')

# Define parameters
max_encoder_length = 30  # Example value, adjust as needed
max_prediction_length = 10  # Example value, adjust as needed
cut_days = 30  # Example value, adjust as needed

# Create training and validation datasets
training_cutoff = df_data["time_idx"].max() - cut_days

training = TimeSeriesDataSet(
    df_data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",
    group_ids=["group_id"],  # Adjust based on your data
    min_encoder_length=1,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["value"],
    static_categoricals=["category"],  # Adjust based on your data
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, df_data, predict=True, stop_randomization=True)

# Create data loaders
batch_size = 64  # Example value, adjust as needed
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size)

# Set up the trainer
early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min")
lr_logger = LearningRateMonitor(logging_interval='step')
logger = TensorBoardLogger("logs/")

trainer = pl.Trainer(
    max_epochs=50,  # Example value, adjust as needed
    callbacks=[early_stop_callback, lr_logger],
    logger=logger,
    accelerator="auto",
)

# Initialize and fit the model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=max_prediction_length,
    loss=QuantileLoss(),
)

trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Save the best model
best_model_path = trainer.checkpoint_callback.best_model_path
tft.save(best_model_path)