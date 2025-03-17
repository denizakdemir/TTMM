#!/usr/bin/env python
# TTML Wine Regression Example (Debug Version)

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Import TTML modules
from tabular_transformer.models import TabularTransformer
from tabular_transformer.models.task_heads import RegressionHead
from tabular_transformer.training import Trainer
from tabular_transformer.utils.config import TransformerConfig, ModelConfig, TaskConfig
from tabular_transformer.data.dataset import TabularDataset

# Import data utilities
from tabular_transformer.examples.data_utils import download_wine_quality_dataset

# Part 1: Wine Quality Prediction
# Download Wine Quality dataset
wine_df = download_wine_quality_dataset(save_csv=False, variant='red')
print("Wine Quality dataset shape:", wine_df.shape)
print("\nFeature types:")
print(wine_df.dtypes)
print("\nQuality score distribution:")
print(wine_df['class'].value_counts().sort_index())

# Identify numeric and categorical columns
numeric_features = wine_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = wine_df.select_dtypes(include=['category', 'object']).columns.tolist()

# Remove target column from features
target_column = 'class'
if target_column in numeric_features:
    numeric_features.remove(target_column)
if target_column in categorical_features:
    categorical_features.remove(target_column)

# Create train/test datasets
train_dataset_wine, test_dataset_wine, _ = TabularDataset.from_dataframe(
    dataframe=wine_df,
    numeric_columns=numeric_features,
    categorical_columns=categorical_features,
    target_columns={'main': [target_column]},
    validation_split=0.2,
    random_state=42
)

# Get feature dimensions from preprocessor
feature_dims = train_dataset_wine.preprocessor.get_feature_dimensions()
numeric_dim = feature_dims['numeric_dim']
categorical_dims = feature_dims['categorical_dims']
categorical_embedding_dims = feature_dims['categorical_embedding_dims']

# Model configuration
config = TransformerConfig(
    embed_dim=64,
    num_heads=4,
    num_layers=3,
    dropout=0.1,
    variational=False
)

# Initialize transformer encoder
encoder_wine = TabularTransformer(
    numeric_dim=numeric_dim,
    categorical_dims=categorical_dims,
    categorical_embedding_dims=categorical_embedding_dims,
    config=config
)

# Initialize regression head
task_head_wine = RegressionHead(
    name='main',  # Must match the task name
    input_dim=64,  # Should match config.embed_dim
    output_dim=1  # Single target value
)

# Create data loaders
train_loader_wine = train_dataset_wine.create_dataloader(batch_size=32, shuffle=True)
test_loader_wine = test_dataset_wine.create_dataloader(batch_size=32, shuffle=False)

# Create a complete ModelConfig with default values
model_config_wine = ModelConfig(
    transformer=config,
    tasks={'main': TaskConfig(
        name='main',
        type='regression',
        output_dim=1,
        target_columns=[target_column],
        weight=1.0
    )},
    learning_rate=1e-3,
    weight_decay=1e-5
)

# Initialize trainer
trainer_wine = Trainer(
    encoder=encoder_wine,
    task_head={'main': task_head_wine},  # Map task head to task name
    config=model_config_wine,  # Pass the configuration
    optimizer=None,  # Will be created by trainer
    device=None  # Will use CUDA if available
)

# Train the model
history_wine = trainer_wine.train(
    train_loader=train_loader_wine,
    val_loader=test_loader_wine,
    num_epochs=5,  # Reduced for faster debugging
    early_stopping_patience=3
)

# Make predictions
print("\nMaking predictions...")
predictions_wine = trainer_wine.predict(test_loader_wine)

# Print prediction structure
print("\nPrediction structure:")
for task_name, task_predictions in predictions_wine.items():
    if task_name != 'latent_representations':
        print(f"Task: {task_name}")
        for key, value in task_predictions.items():
            print(f"  - {key}: {type(value)}, shape: {value.shape}")

# Get predictions for the main task
print("\nExtracting predictions...")
y_pred_wine = predictions_wine['main']['prediction'].numpy()
y_test_wine = test_dataset_wine.targets['main']

print("\nPredictions shape:", y_pred_wine.shape)
print("Target shape:", y_test_wine.shape)

# Calculate metrics
mse = mean_squared_error(y_test_wine, y_pred_wine)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_wine, y_pred_wine)
r2 = r2_score(y_test_wine, y_pred_wine)

print("\nWine Quality Regression Results:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

print("\nDone!")