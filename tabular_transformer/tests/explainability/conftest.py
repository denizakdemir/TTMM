"""
Test fixtures for explainability tests.

This module provides reusable pytest fixtures for testing
explainability features across different task head types.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_regression, make_classification
from typing import Dict, List, Tuple, Optional

from tabular_transformer.utils.config import TransformerConfig
from tabular_transformer.models.transformer_encoder import TabularTransformer
from tabular_transformer.data.preprocess import FeaturePreprocessor
from tabular_transformer.data.dataset import TabularDataset
from tabular_transformer.training.trainer import Trainer
from tabular_transformer.inference.predict import Predictor
from tabular_transformer.models.task_heads import (
    RegressionHead, ClassificationHead, SurvivalHead,
    CompetingRisksHead, CountHead, ClusteringHead
)


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data for testing."""
    # Create synthetic regression data
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrames
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.Series(y, name="target")
    
    # Split into train and test
    train_df = pd.concat([X_df.iloc[:150], y_df.iloc[:150]], axis=1)
    test_df = pd.concat([X_df.iloc[150:], y_df.iloc[150:]], axis=1)
    
    return {
        "feature_names": feature_names,
        "train": train_df,
        "test": test_df,
        "target_column": "target"
    }


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data for testing."""
    # Create synthetic classification data
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrames
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.Series(y, name="target")
    
    # Split into train and test
    train_df = pd.concat([X_df.iloc[:150], y_df.iloc[:150]], axis=1)
    test_df = pd.concat([X_df.iloc[150:], y_df.iloc[150:]], axis=1)
    
    return {
        "feature_names": feature_names,
        "train": train_df,
        "test": test_df,
        "target_column": "target",
        "num_classes": 2
    }


@pytest.fixture
def sample_survival_data():
    """Generate sample survival data for testing."""
    # Create synthetic data with features
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    # Create feature data
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Generate survival times based on features (some features are more important)
    # Base times from exponential distribution
    base_time = np.random.exponential(scale=100, size=n_samples)
    
    # Modify times based on first 3 features
    effect = X[:, 0] * 10 + X[:, 1] * 5 + X[:, 2] * 3
    time = base_time * np.exp(-effect / 20)
    
    # Generate censoring
    censoring_time = np.random.exponential(scale=120, size=n_samples)
    observed_time = np.minimum(time, censoring_time)
    event = (time <= censoring_time).astype(int)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrames
    X_df = pd.DataFrame(X, columns=feature_names)
    time_df = pd.Series(observed_time, name="time")
    event_df = pd.Series(event, name="event")
    
    # Combine into one DataFrame
    full_df = pd.concat([X_df, time_df, event_df], axis=1)
    
    # Split into train and test
    train_df = full_df.iloc[:150]
    test_df = full_df.iloc[150:]
    
    return {
        "feature_names": feature_names,
        "train": train_df,
        "test": test_df,
        "time_column": "time",
        "event_column": "event"
    }


@pytest.fixture
def sample_count_data():
    """Generate sample count data for testing."""
    # Create synthetic data with features
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    # Create feature data
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Generate count target based on features
    # Base counts from Poisson distribution
    lambda_base = 5
    effect = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]
    lambda_adjusted = lambda_base * np.exp(effect)
    counts = np.random.poisson(lambda_adjusted)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrames
    X_df = pd.DataFrame(X, columns=feature_names)
    counts_df = pd.Series(counts, name="count")
    
    # Combine into one DataFrame
    full_df = pd.concat([X_df, counts_df], axis=1)
    
    # Split into train and test
    train_df = full_df.iloc[:150]
    test_df = full_df.iloc[150:]
    
    return {
        "feature_names": feature_names,
        "train": train_df,
        "test": test_df,
        "target_column": "count"
    }


@pytest.fixture
def sample_competing_risks_data():
    """Generate sample competing risks data for testing."""
    # Create synthetic data with features
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    n_risks = 3
    
    # Create feature data
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Generate survival times for each risk
    base_times = []
    for i in range(n_risks):
        # Base times from exponential distribution
        base_time = np.random.exponential(scale=100 + i*20, size=n_samples)
        
        # Modify times based on features (different features affect different risks)
        effect = X[:, i] * 10 + X[:, (i+1) % n_features] * 5
        time = base_time * np.exp(-effect / 20)
        base_times.append(time)
    
    # Determine which risk occurred first and when
    base_times_array = np.column_stack(base_times)
    event_time = np.min(base_times_array, axis=1)
    event_type = np.argmin(base_times_array, axis=1) + 1  # 1-indexed event types
    
    # Generate censoring
    censoring_time = np.random.exponential(scale=150, size=n_samples)
    observed_time = np.minimum(event_time, censoring_time)
    observed_event = np.where(event_time <= censoring_time, event_type, 0)  # 0 means censored
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrames
    X_df = pd.DataFrame(X, columns=feature_names)
    time_df = pd.Series(observed_time, name="time")
    event_df = pd.Series(observed_event, name="event")
    
    # Combine into one DataFrame
    full_df = pd.concat([X_df, time_df, event_df], axis=1)
    
    # Split into train and test
    train_df = full_df.iloc[:150]
    test_df = full_df.iloc[150:]
    
    return {
        "feature_names": feature_names,
        "train": train_df,
        "test": test_df,
        "time_column": "time",
        "event_column": "event",
        "num_risks": n_risks + 1  # +1 for censoring (event=0)
    }


@pytest.fixture
def sample_clustering_data():
    """Generate sample clustering data for testing."""
    # Create synthetic data with features and clusters
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    n_clusters = 3
    
    # Create cluster centers
    centers = np.random.normal(0, 2, size=(n_clusters, n_features))
    
    # Assign samples to clusters
    cluster_assignments = np.random.randint(0, n_clusters, size=n_samples)
    
    # Generate features with noise based on cluster assignments
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        cluster = cluster_assignments[i]
        X[i] = centers[cluster] + np.random.normal(0, 0.5, size=n_features)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split into train and test
    train_df = X_df.iloc[:150].copy()
    test_df = X_df.iloc[150:].copy()
    
    return {
        "feature_names": feature_names,
        "train": train_df,
        "test": test_df,
        "n_clusters": n_clusters
    }


def create_model_and_preprocessor(
    data: Dict,
    task_type: str,
    task_params: Optional[Dict] = None
) -> Tuple[Predictor, FeaturePreprocessor]:
    """
    Create and train a model for the specified task type.
    
    Args:
        data: Dictionary with train and test data
        task_type: Type of task ('regression', 'classification', etc.)
        task_params: Additional parameters for the task head
        
    Returns:
        Tuple of (predictor, preprocessor)
    """
    feature_names = data["feature_names"]
    train_df = data["train"]
    
    # Create preprocessor
    preprocessor = FeaturePreprocessor(
        numeric_columns=feature_names,
        categorical_columns=[]
    )
    preprocessor.fit(train_df)
    
    # Configure transformer
    config = TransformerConfig(
        input_dim=len(feature_names),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    # Create encoder
    encoder = TabularTransformer(
        numeric_dim=len(feature_names),
        categorical_dims={},
        categorical_embedding_dims={},
        config=config
    )
    
    # Create task head based on task type
    if task_type == "regression":
        task_head = RegressionHead(
            name="regression",
            input_dim=32,
            hidden_dims=[16],
            dropout=0.1,
            **task_params or {}
        )
        target_columns = {"regression": [data["target_column"]]}
        
    elif task_type == "classification":
        num_classes = data.get("num_classes", 2)
        task_head = ClassificationHead(
            name="classification",
            input_dim=32,
            num_classes=num_classes,
            hidden_dims=[16],
            dropout=0.1,
            **task_params or {}
        )
        target_columns = {"classification": [data["target_column"]]}
        
    elif task_type == "survival":
        task_head = SurvivalHead(
            name="survival",
            input_dim=32,
            hidden_dims=[16],
            dropout=0.1,
            num_time_bins=20,  # Add required parameter
            **task_params or {}
        )
        target_columns = {"survival": [data["time_column"], data["event_column"]]}
        
    elif task_type == "count":
        task_head = CountHead(
            name="count",
            input_dim=32,
            hidden_dims=[16],
            dropout=0.1,
            **task_params or {}
        )
        target_columns = {"count": [data["target_column"]]}
        
    elif task_type == "competing_risks":
        num_risks = data.get("num_risks", 3)
        task_head = CompetingRisksHead(
            name="competing_risks",
            input_dim=32,
            hidden_dims=[16],
            num_risks=num_risks,
            num_time_bins=20,  # Add required parameter
            dropout=0.1,
            **task_params or {}
        )
        target_columns = {"competing_risks": [data["time_column"], data["event_column"]]}
        
    elif task_type == "clustering":
        n_clusters = data.get("n_clusters", 3)
        task_head = ClusteringHead(
            name="clustering",
            input_dim=32,
            hidden_dims=[16],
            num_clusters=n_clusters,
            dropout=0.1,
            **task_params or {}
        )
        target_columns = {}  # No targets for clustering
        
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Create dataset
    train_dataset = TabularDataset(
        dataframe=train_df,
        numeric_columns=feature_names,
        categorical_columns=[],
        target_columns=target_columns,
        preprocessor=preprocessor,
        fit_preprocessor=False
    )
    
    # Create trainer
    trainer = Trainer(
        encoder=encoder,
        task_head={task_head.name: task_head},
        optimizer=torch.optim.Adam(
            list(encoder.parameters()) + list(task_head.parameters()),
            lr=0.001
        ),
        device="cpu"
    )
    
    # Create data loader
    train_loader = train_dataset.create_dataloader(batch_size=32)
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=None,
        num_epochs=3  # Quick training for tests
    )
    
    # Create predictor
    predictor = Predictor(
        encoder=encoder,
        task_head={task_head.name: task_head},
        preprocessor=preprocessor,
        device="cpu"
    )
    
    return predictor, preprocessor


@pytest.fixture
def regression_model(sample_regression_data):
    """Create a trained regression model."""
    predictor, _ = create_model_and_preprocessor(
        sample_regression_data,
        task_type="regression"
    )
    return predictor


@pytest.fixture
def classification_model(sample_classification_data):
    """Create a trained classification model."""
    predictor, _ = create_model_and_preprocessor(
        sample_classification_data,
        task_type="classification"
    )
    return predictor


@pytest.fixture
def survival_model(sample_survival_data):
    """Create a trained survival model."""
    predictor, _ = create_model_and_preprocessor(
        sample_survival_data,
        task_type="survival"
    )
    return predictor


@pytest.fixture
def count_model(sample_count_data):
    """Create a trained count regression model."""
    predictor, _ = create_model_and_preprocessor(
        sample_count_data,
        task_type="count"
    )
    return predictor


@pytest.fixture
def competing_risks_model(sample_competing_risks_data):
    """Create a trained competing risks model."""
    predictor, _ = create_model_and_preprocessor(
        sample_competing_risks_data,
        task_type="competing_risks"
    )
    return predictor


@pytest.fixture
def clustering_model(sample_clustering_data):
    """Create a trained clustering model."""
    predictor, _ = create_model_and_preprocessor(
        sample_clustering_data,
        task_type="clustering"
    )
    return predictor
