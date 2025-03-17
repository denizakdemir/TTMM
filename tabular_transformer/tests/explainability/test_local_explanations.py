"""
Tests for local explainability methods.

This module tests that local explainability features work with all task heads.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from tabular_transformer.explainability.local_explanations import (
    LIMEExplainer, CounterfactualExplainer
)


def test_lime_explainer_regression(
    regression_model, sample_regression_data
):
    """Test LIME explainer with regression tasks."""
    # Create explainer
    lime_explainer = LIMEExplainer(regression_model)
    
    # Get a sample instance
    sample_instance = sample_regression_data["test"].iloc[0]
    
    # Generate explanation
    explanation = lime_explainer.explain_instance(
        instance=sample_instance,
        task_name="regression",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "feature_importance" in explanation
    assert "intercept" in explanation
    assert "prediction" in explanation
    assert len(explanation["feature_importance"]) <= len(sample_regression_data["feature_names"])
    
    # Test visualization
    fig = lime_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)


def test_lime_explainer_classification(
    classification_model, sample_classification_data
):
    """Test LIME explainer with classification tasks."""
    # Create explainer
    lime_explainer = LIMEExplainer(classification_model)
    
    # Get a sample instance
    sample_instance = sample_classification_data["test"].iloc[0]
    
    # Generate explanation
    explanation = lime_explainer.explain_instance(
        instance=sample_instance,
        task_name="classification",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "feature_importance" in explanation
    assert "intercept" in explanation
    assert "prediction" in explanation
    assert len(explanation["feature_importance"]) <= len(sample_classification_data["feature_names"])
    
    # Test visualization
    fig = lime_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)


def test_lime_explainer_survival(
    survival_model, sample_survival_data
):
    """Test LIME explainer with survival tasks."""
    # Create explainer
    lime_explainer = LIMEExplainer(survival_model)
    
    # Get a sample instance (excluding target columns)
    feature_columns = sample_survival_data["feature_names"]
    sample_instance = sample_survival_data["test"].iloc[0][feature_columns]
    
    # Generate explanation
    explanation = lime_explainer.explain_instance(
        instance=sample_instance,
        task_name="survival",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "feature_importance" in explanation
    assert "intercept" in explanation
    assert "prediction" in explanation
    assert len(explanation["feature_importance"]) <= len(sample_survival_data["feature_names"])
    
    # Test visualization
    fig = lime_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)


def test_lime_explainer_count(
    count_model, sample_count_data
):
    """Test LIME explainer with count regression tasks."""
    # Create explainer
    lime_explainer = LIMEExplainer(count_model)
    
    # Get a sample instance (excluding target column)
    feature_columns = sample_count_data["feature_names"]
    sample_instance = sample_count_data["test"].iloc[0][feature_columns]
    
    # Generate explanation
    explanation = lime_explainer.explain_instance(
        instance=sample_instance,
        task_name="count",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "feature_importance" in explanation
    assert "intercept" in explanation
    assert "prediction" in explanation
    assert len(explanation["feature_importance"]) <= len(sample_count_data["feature_names"])
    
    # Test visualization
    fig = lime_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)


def test_lime_explainer_competing_risks(
    competing_risks_model, sample_competing_risks_data
):
    """Test LIME explainer with competing risks tasks."""
    # Create explainer
    lime_explainer = LIMEExplainer(competing_risks_model)
    
    # Get a sample instance (excluding target columns)
    feature_columns = sample_competing_risks_data["feature_names"]
    sample_instance = sample_competing_risks_data["test"].iloc[0][feature_columns]
    
    # Generate explanation
    explanation = lime_explainer.explain_instance(
        instance=sample_instance,
        task_name="competing_risks",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "feature_importance" in explanation
    assert "intercept" in explanation
    assert "prediction" in explanation
    assert len(explanation["feature_importance"]) <= len(sample_competing_risks_data["feature_names"])
    
    # Test visualization
    fig = lime_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)


def test_lime_explainer_clustering(
    clustering_model, sample_clustering_data
):
    """Test LIME explainer with clustering tasks."""
    # Create explainer
    lime_explainer = LIMEExplainer(clustering_model)
    
    # Get a sample instance
    sample_instance = sample_clustering_data["test"].iloc[0]
    
    # Generate explanation
    explanation = lime_explainer.explain_instance(
        instance=sample_instance,
        task_name="clustering",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "feature_importance" in explanation
    assert "intercept" in explanation
    assert "prediction" in explanation
    assert len(explanation["feature_importance"]) <= len(sample_clustering_data["feature_names"])
    
    # Test visualization
    fig = lime_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)


def test_counterfactual_explainer_regression(
    regression_model, sample_regression_data
):
    """Test counterfactual explainer with regression tasks."""
    # Create explainer
    cf_explainer = CounterfactualExplainer(regression_model)
    
    # Get a sample instance
    sample_instance = sample_regression_data["test"].iloc[0]
    
    # Generate explanation
    explanation = cf_explainer.explain_instance(
        instance=sample_instance,
        task_name="regression",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "counterfactual" in explanation
    assert "distances" in explanation
    assert "original" in explanation
    assert "original_prediction" in explanation
    assert "counterfactual_prediction" in explanation
    
    # Ensure counterfactual is different from original
    assert not pd.Series(explanation["counterfactual"]).equals(pd.Series(explanation["original"]))
    
    # Test visualization
    fig = cf_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)


def test_counterfactual_explainer_classification(
    classification_model, sample_classification_data
):
    """Test counterfactual explainer with classification tasks."""
    # Create explainer
    cf_explainer = CounterfactualExplainer(classification_model)
    
    # Get a sample instance
    sample_instance = sample_classification_data["test"].iloc[0]
    
    # Generate explanation
    explanation = cf_explainer.explain_instance(
        instance=sample_instance,
        task_name="classification",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "counterfactual" in explanation
    assert "distances" in explanation
    assert "original" in explanation
    assert "original_prediction" in explanation
    assert "counterfactual_prediction" in explanation
    
    # Ensure counterfactual is different from original
    assert not pd.Series(explanation["counterfactual"]).equals(pd.Series(explanation["original"]))
    
    # Test visualization
    fig = cf_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)


def test_counterfactual_explainer_survival(
    survival_model, sample_survival_data
):
    """Test counterfactual explainer with survival tasks."""
    # Create explainer
    cf_explainer = CounterfactualExplainer(survival_model)
    
    # Get a sample instance (excluding target columns)
    feature_columns = sample_survival_data["feature_names"]
    sample_instance = sample_survival_data["test"].iloc[0][feature_columns]
    
    # Generate explanation
    explanation = cf_explainer.explain_instance(
        instance=sample_instance,
        task_name="survival",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "counterfactual" in explanation
    assert "distances" in explanation
    assert "original" in explanation
    assert "original_prediction" in explanation
    assert "counterfactual_prediction" in explanation
    
    # Ensure counterfactual is different from original
    assert not pd.Series(explanation["counterfactual"]).equals(pd.Series(explanation["original"]))
    
    # Test visualization
    fig = cf_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)


def test_counterfactual_explainer_count(
    count_model, sample_count_data
):
    """Test counterfactual explainer with count regression tasks."""
    # Create explainer
    cf_explainer = CounterfactualExplainer(count_model)
    
    # Get a sample instance (excluding target column)
    feature_columns = sample_count_data["feature_names"]
    sample_instance = sample_count_data["test"].iloc[0][feature_columns]
    
    # Generate explanation
    explanation = cf_explainer.explain_instance(
        instance=sample_instance,
        task_name="count",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "counterfactual" in explanation
    assert "distances" in explanation
    assert "original" in explanation
    assert "original_prediction" in explanation
    assert "counterfactual_prediction" in explanation
    
    # Ensure counterfactual is different from original
    assert not pd.Series(explanation["counterfactual"]).equals(pd.Series(explanation["original"]))
    
    # Test visualization
    fig = cf_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)


def test_counterfactual_explainer_competing_risks(
    competing_risks_model, sample_competing_risks_data
):
    """Test counterfactual explainer with competing risks tasks."""
    # Create explainer
    cf_explainer = CounterfactualExplainer(competing_risks_model)
    
    # Get a sample instance (excluding target columns)
    feature_columns = sample_competing_risks_data["feature_names"]
    sample_instance = sample_competing_risks_data["test"].iloc[0][feature_columns]
    
    # Generate explanation
    explanation = cf_explainer.explain_instance(
        instance=sample_instance,
        task_name="competing_risks",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "counterfactual" in explanation
    assert "distances" in explanation
    assert "original" in explanation
    assert "original_prediction" in explanation
    assert "counterfactual_prediction" in explanation
    
    # Ensure counterfactual is different from original
    assert not pd.Series(explanation["counterfactual"]).equals(pd.Series(explanation["original"]))
    
    # Test visualization
    fig = cf_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)


def test_counterfactual_explainer_clustering(
    clustering_model, sample_clustering_data
):
    """Test counterfactual explainer with clustering tasks."""
    # Create explainer
    cf_explainer = CounterfactualExplainer(clustering_model)
    
    # Get a sample instance
    sample_instance = sample_clustering_data["test"].iloc[0]
    
    # Generate explanation
    explanation = cf_explainer.explain_instance(
        instance=sample_instance,
        task_name="clustering",
    )
    
    # Verify output structure and content
    assert explanation is not None
    assert "counterfactual" in explanation
    assert "distances" in explanation
    assert "original" in explanation
    assert "original_prediction" in explanation
    assert "counterfactual_prediction" in explanation
    
    # Ensure counterfactual is different from original
    assert not pd.Series(explanation["counterfactual"]).equals(pd.Series(explanation["original"]))
    
    # Test visualization
    fig = cf_explainer.plot_explanation(explanation)
    assert isinstance(fig, plt.Figure)
