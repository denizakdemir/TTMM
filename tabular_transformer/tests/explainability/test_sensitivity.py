"""
Tests for sensitivity analysis methods.

This module tests that sensitivity analysis features work with all task heads.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from tabular_transformer.explainability.sensitivity import SensitivityAnalyzer


def test_sensitivity_regression(
    regression_model, sample_regression_data
):
    """Test sensitivity analyzer with regression tasks."""
    # Create sensitivity analyzer
    sensitivity_analyzer = SensitivityAnalyzer(regression_model)
    
    # Get a sample instance
    sample_instance = sample_regression_data["test"].iloc[0]
    
    # Compute sensitivity
    sensitivity_result = sensitivity_analyzer.compute_sensitivity(
        instance=sample_instance,
        task_name="regression",
        n_samples=50,  # Small number for testing
    )
    
    # Verify output structure and content
    assert sensitivity_result is not None
    assert "feature_sensitivities" in sensitivity_result
    assert "original_prediction" in sensitivity_result
    assert isinstance(sensitivity_result["feature_sensitivities"], dict)
    assert len(sensitivity_result["feature_sensitivities"]) == len(sample_regression_data["feature_names"])
    
    # Verify each feature has correct sensitivity data
    for feature, data in sensitivity_result["feature_sensitivities"].items():
        assert "values" in data
        assert "predictions" in data
        assert len(data["values"]) == len(data["predictions"])
        assert len(data["values"]) > 0
    
    # Test visualization
    fig = sensitivity_analyzer.plot_sensitivity(sensitivity_result)
    assert isinstance(fig, plt.Figure)
    
    # Test tornado plot
    tornado_fig = sensitivity_analyzer.plot_tornado(sensitivity_result)
    assert isinstance(tornado_fig, plt.Figure)


def test_sensitivity_classification(
    classification_model, sample_classification_data
):
    """Test sensitivity analyzer with classification tasks."""
    # Create sensitivity analyzer
    sensitivity_analyzer = SensitivityAnalyzer(classification_model)
    
    # Get a sample instance
    sample_instance = sample_classification_data["test"].iloc[0]
    
    # Compute sensitivity
    sensitivity_result = sensitivity_analyzer.compute_sensitivity(
        instance=sample_instance,
        task_name="classification",
        n_samples=50,  # Small number for testing
    )
    
    # Verify output structure and content
    assert sensitivity_result is not None
    assert "feature_sensitivities" in sensitivity_result
    assert "original_prediction" in sensitivity_result
    assert isinstance(sensitivity_result["feature_sensitivities"], dict)
    assert len(sensitivity_result["feature_sensitivities"]) == len(sample_classification_data["feature_names"])
    
    # Verify each feature has correct sensitivity data
    for feature, data in sensitivity_result["feature_sensitivities"].items():
        assert "values" in data
        assert "predictions" in data
        assert len(data["values"]) == len(data["predictions"])
        assert len(data["values"]) > 0
    
    # Test visualization
    fig = sensitivity_analyzer.plot_sensitivity(sensitivity_result)
    assert isinstance(fig, plt.Figure)
    
    # Test tornado plot
    tornado_fig = sensitivity_analyzer.plot_tornado(sensitivity_result)
    assert isinstance(tornado_fig, plt.Figure)


def test_sensitivity_survival(
    survival_model, sample_survival_data
):
    """Test sensitivity analyzer with survival tasks."""
    # Create sensitivity analyzer
    sensitivity_analyzer = SensitivityAnalyzer(survival_model)
    
    # Get a sample instance (excluding target columns)
    feature_columns = sample_survival_data["feature_names"]
    sample_instance = sample_survival_data["test"].iloc[0][feature_columns]
    
    # Compute sensitivity
    sensitivity_result = sensitivity_analyzer.compute_sensitivity(
        instance=sample_instance,
        task_name="survival",
        n_samples=50,  # Small number for testing
    )
    
    # Verify output structure and content
    assert sensitivity_result is not None
    assert "feature_sensitivities" in sensitivity_result
    assert "original_prediction" in sensitivity_result
    assert isinstance(sensitivity_result["feature_sensitivities"], dict)
    assert len(sensitivity_result["feature_sensitivities"]) == len(sample_survival_data["feature_names"])
    
    # Verify each feature has correct sensitivity data
    for feature, data in sensitivity_result["feature_sensitivities"].items():
        assert "values" in data
        assert "predictions" in data
        assert len(data["values"]) == len(data["predictions"])
        assert len(data["values"]) > 0
    
    # Test visualization
    fig = sensitivity_analyzer.plot_sensitivity(sensitivity_result)
    assert isinstance(fig, plt.Figure)
    
    # Test tornado plot
    tornado_fig = sensitivity_analyzer.plot_tornado(sensitivity_result)
    assert isinstance(tornado_fig, plt.Figure)


def test_sensitivity_count(
    count_model, sample_count_data
):
    """Test sensitivity analyzer with count regression tasks."""
    # Create sensitivity analyzer
    sensitivity_analyzer = SensitivityAnalyzer(count_model)
    
    # Get a sample instance (excluding target column)
    feature_columns = sample_count_data["feature_names"]
    sample_instance = sample_count_data["test"].iloc[0][feature_columns]
    
    # Compute sensitivity
    sensitivity_result = sensitivity_analyzer.compute_sensitivity(
        instance=sample_instance,
        task_name="count",
        n_samples=50,  # Small number for testing
    )
    
    # Verify output structure and content
    assert sensitivity_result is not None
    assert "feature_sensitivities" in sensitivity_result
    assert "original_prediction" in sensitivity_result
    assert isinstance(sensitivity_result["feature_sensitivities"], dict)
    assert len(sensitivity_result["feature_sensitivities"]) == len(sample_count_data["feature_names"])
    
    # Verify each feature has correct sensitivity data
    for feature, data in sensitivity_result["feature_sensitivities"].items():
        assert "values" in data
        assert "predictions" in data
        assert len(data["values"]) == len(data["predictions"])
        assert len(data["values"]) > 0
    
    # Test visualization
    fig = sensitivity_analyzer.plot_sensitivity(sensitivity_result)
    assert isinstance(fig, plt.Figure)
    
    # Test tornado plot
    tornado_fig = sensitivity_analyzer.plot_tornado(sensitivity_result)
    assert isinstance(tornado_fig, plt.Figure)


def test_sensitivity_competing_risks(
    competing_risks_model, sample_competing_risks_data
):
    """Test sensitivity analyzer with competing risks tasks."""
    # Create sensitivity analyzer
    sensitivity_analyzer = SensitivityAnalyzer(competing_risks_model)
    
    # Get a sample instance (excluding target columns)
    feature_columns = sample_competing_risks_data["feature_names"]
    sample_instance = sample_competing_risks_data["test"].iloc[0][feature_columns]
    
    # Compute sensitivity
    sensitivity_result = sensitivity_analyzer.compute_sensitivity(
        instance=sample_instance,
        task_name="competing_risks",
        n_samples=50,  # Small number for testing
    )
    
    # Verify output structure and content
    assert sensitivity_result is not None
    assert "feature_sensitivities" in sensitivity_result
    assert "original_prediction" in sensitivity_result
    assert isinstance(sensitivity_result["feature_sensitivities"], dict)
    assert len(sensitivity_result["feature_sensitivities"]) == len(sample_competing_risks_data["feature_names"])
    
    # Verify each feature has correct sensitivity data
    for feature, data in sensitivity_result["feature_sensitivities"].items():
        assert "values" in data
        assert "predictions" in data
        assert len(data["values"]) == len(data["predictions"])
        assert len(data["values"]) > 0
    
    # Test visualization
    fig = sensitivity_analyzer.plot_sensitivity(sensitivity_result)
    assert isinstance(fig, plt.Figure)
    
    # Test tornado plot
    tornado_fig = sensitivity_analyzer.plot_tornado(sensitivity_result)
    assert isinstance(tornado_fig, plt.Figure)


def test_sensitivity_clustering(
    clustering_model, sample_clustering_data
):
    """Test sensitivity analyzer with clustering tasks."""
    # Create sensitivity analyzer
    sensitivity_analyzer = SensitivityAnalyzer(clustering_model)
    
    # Get a sample instance
    sample_instance = sample_clustering_data["test"].iloc[0]
    
    # Compute sensitivity
    sensitivity_result = sensitivity_analyzer.compute_sensitivity(
        instance=sample_instance,
        task_name="clustering",
        n_samples=50,  # Small number for testing
    )
    
    # Verify output structure and content
    assert sensitivity_result is not None
    assert "feature_sensitivities" in sensitivity_result
    assert "original_prediction" in sensitivity_result
    assert isinstance(sensitivity_result["feature_sensitivities"], dict)
    assert len(sensitivity_result["feature_sensitivities"]) == len(sample_clustering_data["feature_names"])
    
    # Verify each feature has correct sensitivity data
    for feature, data in sensitivity_result["feature_sensitivities"].items():
        assert "values" in data
        assert "predictions" in data
        assert len(data["values"]) == len(data["predictions"])
        assert len(data["values"]) > 0
    
    # Test visualization
    fig = sensitivity_analyzer.plot_sensitivity(sensitivity_result)
    assert isinstance(fig, plt.Figure)
    
    # Test tornado plot
    tornado_fig = sensitivity_analyzer.plot_tornado(sensitivity_result)
    assert isinstance(tornado_fig, plt.Figure)
