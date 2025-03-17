#!/usr/bin/env python
# TTML Clustering Examples - Simplified

import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Import TTML modules
from tabular_transformer.models import TabularTransformer
from tabular_transformer.models.task_heads import ClusteringHead
from tabular_transformer.utils.config import TransformerConfig
from tabular_transformer.data.dataset import TabularDataset

# Import data utilities
from data_utils import download_wine_quality_dataset

# Download Wine Quality dataset
wine_df = download_wine_quality_dataset(save_csv=False, variant='red')
print("Wine Quality dataset shape:", wine_df.shape)
print("\nFeature types:")
print(wine_df.dtypes)

# Identify numeric and categorical columns
df_numeric_features = wine_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
df_categorical_features = wine_df.select_dtypes(include=['category', 'object']).columns.tolist()

# Remove quality column from features
quality_column = 'class'  # The target column is named 'class' in the dataset
if quality_column in df_numeric_features:
    df_numeric_features.remove(quality_column)
if quality_column in df_categorical_features:
    df_categorical_features.remove(quality_column)

# Create dataset with dummy targets (we'll ignore them during clustering)
train_dataset, test_dataset, _ = TabularDataset.from_dataframe(
    dataframe=wine_df,
    numeric_columns=df_numeric_features,
    categorical_columns=df_categorical_features,
    target_columns={'main': [quality_column]},  # We need to provide a target for the API to work
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

# Create data loaders
train_loader = train_dataset.create_dataloader(batch_size=32, shuffle=True)
test_loader = test_dataset.create_dataloader(batch_size=32, shuffle=False)

# Simple manual approach since training currently needs targets:
print("\nPerforming manual clustering:")

# Get all features in one batch
all_features = []

with torch.no_grad():
    for batch in train_loader:
        # Move inputs to device
        batch_numeric_features = batch["numeric_features"]
        batch_numeric_mask = batch["numeric_mask"]
        batch_categorical_features = batch["categorical_features"]
        batch_categorical_mask = batch["categorical_mask"]
        
        # Get encoder embeddings
        embeddings = encoder(
            numeric_features=batch_numeric_features, 
            numeric_mask=batch_numeric_mask,
            categorical_features=batch_categorical_features,
            categorical_mask=batch_categorical_mask
        )
        
        all_features.append(embeddings.cpu().numpy())

# Concatenate all batches
all_embeddings = np.vstack(all_features)

# Use sklearn KMeans for clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_assignments = kmeans.fit_predict(all_embeddings)

# Reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(all_embeddings)

# Plot clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_assignments, cmap='viridis')
plt.colorbar(scatter)
plt.title('Wine Clusters in Latent Space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.show()

# Create a new DataFrame just for the cluster results
# We know the number of training samples from the TabularDataset log output
train_sample_count = len(train_dataset)  # Should be 1280 samples
wine_df_train = wine_df.iloc[:train_sample_count].copy()  # Take the first 1280 samples
# Ensure the length matches
print(f"Training samples: {len(wine_df_train)}, Cluster assignments: {len(cluster_assignments)}")
if len(wine_df_train) != len(cluster_assignments):
    # Trim to the smaller size if needed
    min_len = min(len(wine_df_train), len(cluster_assignments))
    wine_df_train = wine_df_train.iloc[:min_len].copy()
    cluster_assignments = cluster_assignments[:min_len]
wine_df_train['Cluster'] = cluster_assignments

# Calculate cluster statistics
cluster_stats = wine_df_train.groupby('Cluster')[df_numeric_features].agg(['mean', 'std']).round(2)

print("\nCluster Statistics:")
print(cluster_stats)

# Plot feature distributions by cluster for key features
# Pick a few important features
key_features = ['alcohol', 'volatile_acidity', 'sulphates', 'pH']
# Make sure all features exist in the dataframe
key_features = [f for f in key_features if f in df_numeric_features]
if key_features:
    plt.figure(figsize=(15, 12))
    for i, feature in enumerate(key_features, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=wine_df_train, x='Cluster', y=feature)
        plt.title(f'{feature} by Cluster')
    plt.tight_layout()
    plt.show()
else:
    print("No key features found for visualization")

# Compare clusters with the actual wine quality
quality_by_cluster = pd.crosstab(wine_df_train['Cluster'], wine_df_train[quality_column])
print("\nWine Quality Distribution by Cluster:")
print(quality_by_cluster)

# Plot quality distribution by cluster
plt.figure(figsize=(12, 6))
quality_by_cluster.plot(kind='bar', stacked=True)
plt.title('Wine Quality Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Wine Quality')
plt.tight_layout()
plt.show()

print("\nDone! This simplified clustering example demonstrates how to extract embeddings")
print("from the TabularTransformer and apply KMeans clustering to them.")
print("The complete trainer-integrated clustering is still under development.")