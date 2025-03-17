#!/usr/bin/env python
# TTML Classification Examples

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import TTML modules
from tabular_transformer.models import TabularTransformer
from tabular_transformer.models.task_heads import ClassificationHead
from tabular_transformer.training import Trainer
from tabular_transformer.inference import predict
from tabular_transformer.explainability import global_explanations, local_explanations
from tabular_transformer.utils.config import TransformerConfig, ModelConfig, TaskConfig
from tabular_transformer.data.dataset import TabularDataset

# Import data utilities
from data_utils import download_adult_dataset, download_titanic_dataset

# Part 1: Adult Income Classification
# Download Adult dataset
adult_df = download_adult_dataset(save_csv=False)
print("Adult dataset shape:", adult_df.shape)
print("\nFeature types:")
print(adult_df.dtypes)
print("\nClass distribution:")
print(adult_df['class'].value_counts(normalize=True))

# Identify numeric and categorical columns
numeric_features = adult_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = adult_df.select_dtypes(include=['object']).columns.tolist()

# Remove target column from features
target_column = 'class'
if target_column in numeric_features:
    numeric_features.remove(target_column)
if target_column in categorical_features:
    categorical_features.remove(target_column)

# Preprocess the class column to convert string labels to numeric
print("Original class values:", adult_df['class'].unique())
adult_df['class'] = adult_df['class'].map({'>50K': 1, '<=50K': 0})
print("Converted class values:", adult_df['class'].unique())

train_dataset_adult, test_dataset_adult, _ = TabularDataset.from_dataframe(
    dataframe=adult_df,
    numeric_columns=numeric_features,
    categorical_columns=categorical_features,
    target_columns={'main': [target_column]},
    validation_split=0.2,
    random_state=42
)

# Get feature dimensions from preprocessor
feature_dims = train_dataset_adult.preprocessor.get_feature_dimensions()
numeric_dim = feature_dims['numeric_dim']
categorical_dims = feature_dims['categorical_dims']
categorical_embedding_dims = feature_dims['categorical_embedding_dims']

# Model configuration
config = TransformerConfig(
    embed_dim=128,
    num_heads=8,
    num_layers=4,
    dropout=0.2,
    variational=False
)

# Initialize transformer encoder
encoder_adult = TabularTransformer(
    numeric_dim=numeric_dim,
    categorical_dims=categorical_dims,
    categorical_embedding_dims=categorical_embedding_dims,
    config=config
)

# Initialize classification head
task_head_adult = ClassificationHead(
    name="main",  # Task name should match the key in target_columns
    input_dim=128,  # Should match config.embed_dim
    num_classes=2  # Binary classification for income
)

# Create data loaders
train_loader_adult = train_dataset_adult.create_dataloader(batch_size=64, shuffle=True)
test_loader_adult = test_dataset_adult.create_dataloader(batch_size=64, shuffle=False)

# Create a complete ModelConfig with default values
model_config_adult = ModelConfig(
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
trainer_adult = Trainer(
    encoder=encoder_adult,
    task_head={'main': task_head_adult},  # Map task head to task name
    config=model_config_adult,  # Pass the configuration
    optimizer=None,  # Will be created by trainer
    device=None  # Will use CUDA if available
)

# Train the model
history_adult = trainer_adult.train(
    train_loader=train_loader_adult,
    val_loader=test_loader_adult,
    num_epochs=20,
    early_stopping_patience=3
)

# Make predictions
predictions_adult = trainer_adult.predict(test_loader_adult)

# Get predictions for the main task
y_pred_adult = predictions_adult['main']['predicted_class'].numpy()
y_test_adult = test_dataset_adult.targets['main']

# Print metrics
print("Adult Income Classification Results:")
print(f"Accuracy: {accuracy_score(y_test_adult, y_pred_adult):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_adult, y_pred_adult))

# Feature Importance Analysis
# Create a Predictor instance for the adult dataset
from tabular_transformer.inference.predict import Predictor

# Create predictor for feature importance analysis
predictor_adult = Predictor(
    encoder=encoder_adult,
    task_head={'main': task_head_adult},
    preprocessor=test_dataset_adult.preprocessor,
    device=None  # Will use the same device as the model
)

# Initialize permutation importance explainer
permutation_explainer = global_explanations.PermutationImportance(
    predictor=predictor_adult,
    feature_names=numeric_features + categorical_features
)

# Skip feature importance calculation for now as it requires additional implementation
print("\nNote: Feature importance calculation skipped in this version.")

# Part 2: Titanic Survival Classification
# Download Titanic dataset
titanic_df = download_titanic_dataset(save_csv=False)
print("Titanic dataset shape:", titanic_df.shape)
print("\nFeature types:")
print(titanic_df.dtypes)
print("\nSurvival distribution:")
print(titanic_df['survived'].value_counts(normalize=True))

# Identify numeric and categorical columns
numeric_features = titanic_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = titanic_df.select_dtypes(include=['object']).columns.tolist()

# Remove target column from features
target_column = 'survived'
if target_column in numeric_features:
    numeric_features.remove(target_column)
if target_column in categorical_features:
    categorical_features.remove(target_column)

# Create train/test datasets
train_dataset_titanic, test_dataset_titanic, _ = TabularDataset.from_dataframe(
    dataframe=titanic_df,
    numeric_columns=numeric_features,
    categorical_columns=categorical_features,
    target_columns={'main': [target_column]},
    validation_split=0.2,
    random_state=42
)

# Get feature dimensions from preprocessor
feature_dims = train_dataset_titanic.preprocessor.get_feature_dimensions()
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
encoder_titanic = TabularTransformer(
    numeric_dim=numeric_dim,
    categorical_dims=categorical_dims,
    categorical_embedding_dims=categorical_embedding_dims,
    config=config
)

# Initialize classification head
task_head_titanic = ClassificationHead(
    name="main",  # Task name should match the key in target_columns
    input_dim=64,  # Should match config.embed_dim
    num_classes=2  # Binary classification for survival
)

# Create data loaders
train_loader_titanic = train_dataset_titanic.create_dataloader(batch_size=32, shuffle=True)
test_loader_titanic = test_dataset_titanic.create_dataloader(batch_size=32, shuffle=False)

# Create a complete ModelConfig with default values
model_config_titanic = ModelConfig(
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
trainer_titanic = Trainer(
    encoder=encoder_titanic,
    task_head={'main': task_head_titanic},  # Map task head to task name
    config=model_config_titanic,  # Pass the configuration
    optimizer=None,  # Will be created by trainer
    device=None  # Will use CUDA if available
)

# Train the model
history_titanic = trainer_titanic.train(
    train_loader=train_loader_titanic,
    val_loader=test_loader_titanic,
    num_epochs=15,
    early_stopping_patience=3
)

# Make predictions
predictions_titanic = trainer_titanic.predict(test_loader_titanic)

# Get predictions for the main task
y_pred_titanic = predictions_titanic['main']['predicted_class'].numpy()
y_test_titanic = test_dataset_titanic.targets['main']

print("Titanic Survival Classification Results:")
print(f"Accuracy: {accuracy_score(y_test_titanic, y_pred_titanic):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_titanic, y_pred_titanic))

# Local Explanations
# Create a Predictor for the Titanic dataset
predictor_titanic = Predictor(
    encoder=encoder_titanic,
    task_head={'main': task_head_titanic},
    preprocessor=test_dataset_titanic.preprocessor,
    device=None  # Will use the same device as the model
)

# Initialize LIME explainer
lime_explainer = local_explanations.LIMEExplainer(
    predictor=predictor_titanic,
    feature_names=numeric_features + categorical_features
)

# Get local explanations for a few examples
print("\nLocal explanations for selected examples:")
sample_indices = np.random.choice(len(test_dataset_titanic), 3, replace=False)
for i, idx in enumerate(sample_indices):
    # Get the instance as a pandas Series
    instance = test_dataset_titanic.data.iloc[idx]
    
    # Get LIME explanation
    explanation = lime_explainer.explain_instance(
        instance=instance,
        task_name='main'
    )
    
    print(f"\nExample {i+1}:")
    print(f"True class: {y_test_titanic[idx]}")
    print(f"Predicted class: {y_pred_titanic[idx]}")
    print("\nFeature contributions (top features):")
    for feature, contribution in explanation['feature_contributions'].items():
        print(f"{feature}: {contribution:.4f}")
