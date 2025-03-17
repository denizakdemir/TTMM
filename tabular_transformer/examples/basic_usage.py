#!/usr/bin/env python
# TTML Basic Usage Example with Titanic dataset

import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report

# Import TTML modules
from tabular_transformer.models import TabularTransformer
from tabular_transformer.models.task_heads import ClassificationHead
from tabular_transformer.training import Trainer
from tabular_transformer.inference import predict
from tabular_transformer.utils.config import TransformerConfig, ModelConfig, TaskConfig
from tabular_transformer.data.dataset import TabularDataset

# Import data utilities
from data_utils import download_titanic_dataset, prepare_dataset

# 1. Load and Preprocess Data
# Download Titanic dataset
df = download_titanic_dataset(save_csv=False)
print("Dataset shape:", df.shape)
print("\nFeature types:")
print(df.dtypes)

# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove target column from features
target_column = 'survived'
if target_column in numeric_features:
    numeric_features.remove(target_column)
if target_column in categorical_features:
    categorical_features.remove(target_column)

# Create train/test datasets
train_dataset, test_dataset, _ = TabularDataset.from_dataframe(
    dataframe=df,
    numeric_columns=numeric_features,
    categorical_columns=categorical_features,
    target_columns={'main': [target_column]},
    validation_split=0.2,
    random_state=42
)

# 2. Configure and Initialize Model
# Get feature dimensions from preprocessor
feature_dims = train_dataset.preprocessor.get_feature_dimensions()
numeric_dim = feature_dims['numeric_dim']
categorical_dims = feature_dims['categorical_dims']
categorical_embedding_dims = feature_dims['categorical_embedding_dims']

# Model configuration
config = TransformerConfig(
    embed_dim=64,
    num_heads=4,
    num_layers=2,
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

# Initialize classification head
task_head = ClassificationHead(
    name="main",  # Task name should match the key in target_columns
    input_dim=64,  # Should match config.embed_dim
    num_classes=2  # Binary classification for survival
)

# 3. Train the Model
# Create data loaders
train_loader = train_dataset.create_dataloader(batch_size=32, shuffle=True)
test_loader = test_dataset.create_dataloader(batch_size=32, shuffle=False)

# Create a complete ModelConfig with default values
model_config = ModelConfig(
    transformer=config,
    tasks={'main': TaskConfig(
        name='main',
        type='classification',
        output_dim=2,
        target_columns=[target_column],
        weight=1.0
    )},
    learning_rate=1e-3,
    weight_decay=1e-5
)

# Initialize trainer
trainer = Trainer(
    encoder=encoder,
    task_head={'main': task_head},  # Map task head to task name
    config=model_config,  # Pass the configuration
    optimizer=None,  # Will be created by trainer
    device=None  # Will use CUDA if available
)

# Train the model
history = trainer.train(
    train_loader=train_loader,
    val_loader=test_loader,
    num_epochs=10,
    early_stopping_patience=3
)

# 4. Make Predictions
# Make predictions
predictions = trainer.predict(test_loader)

# Get predictions for the main task
y_pred = predictions['main']['predicted_class'].numpy()
y_test = test_dataset.targets['main']

# 5. Evaluate Results
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
