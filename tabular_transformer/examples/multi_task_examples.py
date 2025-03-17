#!/usr/bin/env python
# TTML Multi-Task Learning Examples (Simplified)

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Import TTML modules
from tabular_transformer.models import TabularTransformer
from tabular_transformer.models.task_heads import (
    ClassificationHead,
    RegressionHead
)
from tabular_transformer.training import Trainer
from tabular_transformer.utils.config import TransformerConfig, ModelConfig, TaskConfig
from tabular_transformer.data.dataset import TabularDataset

# Import data utilities
from tabular_transformer.examples.data_utils import download_wine_quality_dataset

# Part 1: Wine Quality Regression
print("### Wine Quality Regression Example ###")
wine_df = download_wine_quality_dataset(save_csv=False, variant='red')

# Handle different column names
if 'class' in wine_df.columns:
    wine_df = wine_df.rename(columns={'class': 'quality'})
elif 'Class' in wine_df.columns:
    wine_df = wine_df.rename(columns={'Class': 'quality'})

print("Wine Quality dataset shape:", wine_df.shape)
print("\nFeature types:")
print(wine_df.dtypes)

# Identify numeric and categorical columns
numeric_features = wine_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = wine_df.select_dtypes(include=['category', 'object']).columns.tolist()

# Remove target column from features
quality_column = 'quality'
if quality_column in numeric_features:
    numeric_features.remove(quality_column)
if quality_column in categorical_features:
    categorical_features.remove(quality_column)

# Create train/test datasets with quality as target
train_dataset, test_dataset, _ = TabularDataset.from_dataframe(
    dataframe=wine_df,
    numeric_columns=numeric_features,
    categorical_columns=categorical_features,
    target_columns={'main': [quality_column]},
    validation_split=0.2,
    random_state=42
)

# Get feature dimensions from preprocessor
feature_dims = train_dataset.preprocessor.get_feature_dimensions()
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
encoder = TabularTransformer(
    numeric_dim=numeric_dim,
    categorical_dims=categorical_dims,
    categorical_embedding_dims=categorical_embedding_dims,
    config=config
)

# Initialize regression head
regression_head = RegressionHead(
    name='main',
    input_dim=64,
    output_dim=1
)

# Create data loaders
train_loader = train_dataset.create_dataloader(batch_size=32, shuffle=True)
test_loader = test_dataset.create_dataloader(batch_size=32, shuffle=False)

# Create ModelConfig
model_config = ModelConfig(
    transformer=config,
    tasks={'main': TaskConfig(
        name='main',
        type='regression',
        output_dim=1,
        target_columns=[quality_column],
        weight=1.0
    )},
    learning_rate=1e-3,
    weight_decay=1e-5
)

# Initialize trainer
trainer = Trainer(
    encoder=encoder,
    task_head={'main': regression_head},
    config=model_config,
    device=None
)

# Train model
print("\nTraining regression model...")
history = trainer.train(
    train_loader=train_loader,
    val_loader=test_loader,
    num_epochs=10,
    early_stopping_patience=3
)

# Make predictions
predictions = trainer.predict(test_loader)

# Get predictions
y_pred = predictions['main']['prediction'].numpy()
y_test = test_dataset.targets['main']

# Ensure consistent dimensionality
if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
    y_pred = y_pred.flatten()
if len(y_test.shape) > 1 and y_test.shape[1] == 1:
    y_test = y_test.flatten()

# Convert tensors to numpy if needed
if isinstance(y_test, torch.Tensor):
    y_test = y_test.cpu().numpy()

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nRegression Results:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Part 2: Transfer Learning to a Smaller Dataset
print("\n### Transfer Learning Example ###")

# Create a smaller dataset (10% of original)
n_samples = len(train_dataset) // 10
small_indices = np.random.choice(len(train_dataset.data), n_samples, replace=False)
small_df = train_dataset.data.iloc[small_indices].copy()

# Create small dataset
small_dataset = TabularDataset(
    dataframe=small_df,
    numeric_columns=numeric_features,
    categorical_columns=categorical_features,
    target_columns={'main': [quality_column]},
    preprocessor=train_dataset.preprocessor  # Use same preprocessor
)

print(f"Small dataset size: {len(small_dataset)} samples")

# Train a model from scratch on the small dataset
scratch_encoder = TabularTransformer(
    numeric_dim=numeric_dim,
    categorical_dims=categorical_dims,
    categorical_embedding_dims=categorical_embedding_dims,
    config=config
)

scratch_head = RegressionHead(
    name='main',
    input_dim=64,
    output_dim=1
)

# Create data loader for small dataset
small_loader = small_dataset.create_dataloader(batch_size=16, shuffle=True)

# Initialize trainer for scratch model
scratch_trainer = Trainer(
    encoder=scratch_encoder,
    task_head={'main': scratch_head},
    config=model_config,
    device=None
)

# Train from scratch
print("\nTraining model from scratch on small dataset...")
scratch_history = scratch_trainer.train(
    train_loader=small_loader,
    val_loader=test_loader,
    num_epochs=10,
    early_stopping_patience=3
)

# Make predictions
scratch_predictions = scratch_trainer.predict(test_loader)

# Get predictions
scratch_pred = scratch_predictions['main']['prediction'].numpy()

# Ensure consistent dimensionality
if len(scratch_pred.shape) > 1 and scratch_pred.shape[1] == 1:
    scratch_pred = scratch_pred.flatten()

# Calculate metrics
scratch_mse = mean_squared_error(y_test, scratch_pred)
scratch_rmse = np.sqrt(scratch_mse)
scratch_r2 = r2_score(y_test, scratch_pred)

print("\nTraining from Scratch Results (Small Dataset):")
print(f"RMSE: {scratch_rmse:.4f}")
print(f"R² Score: {scratch_r2:.4f}")

# Use the pre-trained model for fine-tuning
print("\nFine-tuning pre-trained model on small dataset...")

# Make a copy of the trained encoder
finetune_encoder = TabularTransformer(
    numeric_dim=numeric_dim,
    categorical_dims=categorical_dims,
    categorical_embedding_dims=categorical_embedding_dims,
    config=config
)
# Copy weights from the trained encoder
finetune_encoder.load_state_dict(encoder.state_dict())

# Freeze some encoder layers (demonstration)
# In a real implementation, you might freeze specific layers
for name, param in finetune_encoder.named_parameters():
    if 'embedding' in name or 'input_projection' in name:
        param.requires_grad = False

finetune_head = RegressionHead(
    name='main',
    input_dim=64,
    output_dim=1
)

# Initialize trainer for fine-tuning
finetune_trainer = Trainer(
    encoder=finetune_encoder,
    task_head={'main': finetune_head},
    config=model_config,
    device=None
)

# Fine-tune
finetune_history = finetune_trainer.train(
    train_loader=small_loader,
    val_loader=test_loader,
    num_epochs=10,
    early_stopping_patience=3
)

# Make predictions
finetune_predictions = finetune_trainer.predict(test_loader)

# Get predictions
finetune_pred = finetune_predictions['main']['prediction'].numpy()

# Ensure consistent dimensionality
if len(finetune_pred.shape) > 1 and finetune_pred.shape[1] == 1:
    finetune_pred = finetune_pred.flatten()

# Calculate metrics
finetune_mse = mean_squared_error(y_test, finetune_pred)
finetune_rmse = np.sqrt(finetune_mse)
finetune_r2 = r2_score(y_test, finetune_pred)

print("\nFine-tuning Results:")
print(f"RMSE: {finetune_rmse:.4f}")
print(f"R² Score: {finetune_r2:.4f}")

# Compare results
print("\nComparison:")
print(f"Full Dataset Training: RMSE = {rmse:.4f}, R² = {r2:.4f}")
print(f"From Scratch (Small): RMSE = {scratch_rmse:.4f}, R² = {scratch_r2:.4f}")
print(f"Fine-tuned (Small): RMSE = {finetune_rmse:.4f}, R² = {finetune_r2:.4f}")

print("\nNOTE: This is a simplified version of multi-task learning and transfer learning.")
print("For true multi-task learning, you would need multiple targets and would use MultiTaskHead.")