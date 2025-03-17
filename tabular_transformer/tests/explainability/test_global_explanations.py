"""
Tests for global explainability methods.

This module tests that global explainability features work with all task heads.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from tabular_transformer.explainability.global_explanations import (
    GlobalExplainer, PermutationImportance
)


def test_permutation_importance_regression(
    regression_model, sample_regression_data
):
    """Test permutation importance with regression tasks."""
    # Create explainer
    perm_importance = PermutationImportance(regression_model)
    
    # Calculate importance
    importance = perm_importance.compute_importance(
        data=sample_regression_data["test"],
        target_columns={"regression": "target"},
        task_names=["regression"],
    )
    
    # Verify output
    assert "regression" in importance
    assert isinstance(importance["regression"], np.ndarray)
    assert len(importance["regression"]) == len(sample_regression_data["feature_names"])
    
    # Test plotting
    global_explainer = GlobalExplainer(regression_model)
    fig = global_explainer.plot_feature_importance(importance)
    assert isinstance(fig, plt.Figure)


def test_permutation_importance_classification(
    classification_model, sample_classification_data
):
    """Test permutation importance with classification tasks."""
    # Create explainer
    perm_importance = PermutationImportance(classification_model)
    
    # Calculate importance
    importance = perm_importance.compute_importance(
        data=sample_classification_data["test"],
        target_columns={"classification": "target"},
        task_names=["classification"],
    )
    
    # Verify output
    assert "classification" in importance
    assert isinstance(importance["classification"], np.ndarray)
    assert len(importance["classification"]) == len(sample_classification_data["feature_names"])
    
    # Test plotting
    global_explainer = GlobalExplainer(classification_model)
    fig = global_explainer.plot_feature_importance(importance)
    assert isinstance(fig, plt.Figure)


def test_permutation_importance_survival(
    survival_model, sample_survival_data
):
    """Test permutation importance with survival tasks."""
    # Create explainer
    perm_importance = PermutationImportance(survival_model)
    
    # Calculate importance
    importance = perm_importance.compute_importance(
        data=sample_survival_data["test"],
        target_columns={
            "survival": [
                sample_survival_data["time_column"],
                sample_survival_data["event_column"]
            ]
        },
        task_names=["survival"],
    )
    
    # Verify output
    assert "survival" in importance
    assert isinstance(importance["survival"], np.ndarray)
    assert len(importance["survival"]) == len(sample_survival_data["feature_names"])
    
    # Test plotting
    global_explainer = GlobalExplainer(survival_model)
    fig = global_explainer.plot_feature_importance(importance)
    assert isinstance(fig, plt.Figure)


def test_permutation_importance_count(
    count_model, sample_count_data
):
    """Test permutation importance with count regression tasks."""
    # Create explainer
    perm_importance = PermutationImportance(count_model)
    
    # Calculate importance
    importance = perm_importance.compute_importance(
        data=sample_count_data["test"],
        target_columns={"count": "count"},
        task_names=["count"],
    )
    
    # Verify output
    assert "count" in importance
    assert isinstance(importance["count"], np.ndarray)
    assert len(importance["count"]) == len(sample_count_data["feature_names"])
    
    # Test plotting
    global_explainer = GlobalExplainer(count_model)
    fig = global_explainer.plot_feature_importance(importance)
    assert isinstance(fig, plt.Figure)


def test_permutation_importance_competing_risks(
    competing_risks_model, sample_competing_risks_data
):
    """Test permutation importance with competing risks tasks."""
    # Create explainer
    perm_importance = PermutationImportance(competing_risks_model)
    
    # Calculate importance
    importance = perm_importance.compute_importance(
        data=sample_competing_risks_data["test"],
        target_columns={
            "competing_risks": [
                sample_competing_risks_data["time_column"],
                sample_competing_risks_data["event_column"]
            ]
        },
        task_names=["competing_risks"],
    )
    
    # Verify output
    assert "competing_risks" in importance
    assert isinstance(importance["competing_risks"], np.ndarray)
    assert len(importance["competing_risks"]) == len(sample_competing_risks_data["feature_names"])
    
    # Test plotting
    global_explainer = GlobalExplainer(competing_risks_model)
    fig = global_explainer.plot_feature_importance(importance)
    assert isinstance(fig, plt.Figure)


def test_permutation_importance_clustering(
    clustering_model, sample_clustering_data
):
    """Test permutation importance with clustering tasks."""
    # Create explainer
    perm_importance = PermutationImportance(clustering_model)
    
    # For clustering, we don't have a ground truth target, so we need a different approach
    # We can use cluster stability or silhouette score as the performance metric
    # Here we'll test if the API works without actual evaluation
    
    # Calculate importance (this may fail if the implementation doesn't handle clustering)
    importance = perm_importance.compute_importance(
        data=sample_clustering_data["test"],
        target_columns={},  # No target columns for clustering
        task_names=["clustering"],
    )
    
    # Verify output
    assert "clustering" in importance
    assert isinstance(importance["clustering"], np.ndarray)
    assert len(importance["clustering"]) == len(sample_clustering_data["feature_names"])
    
    # Test plotting
    global_explainer = GlobalExplainer(clustering_model)
    fig = global_explainer.plot_feature_importance(importance)
    assert isinstance(fig, plt.Figure)
