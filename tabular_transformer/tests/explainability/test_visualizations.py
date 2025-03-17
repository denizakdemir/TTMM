"""
Tests for visualization methods.

This module tests that visualization features work with all task heads.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from tabular_transformer.explainability.visualizations import (
    PDPlot, ICEPlot, CalibrationPlot
)


def test_pdp_regression(
    regression_model, sample_regression_data
):
    """Test partial dependence plots with regression tasks."""
    # Create PDP generator
    pd_plot = PDPlot(regression_model)
    
    # Get a feature to visualize
    feature = sample_regression_data["feature_names"][0]
    
    # Generate PDP
    pdp_result = pd_plot.compute_partial_dependence(
        data=sample_regression_data["test"],
        feature=feature,
        task_name="regression",
    )
    
    # Verify output structure and content
    assert pdp_result is not None
    assert "feature" in pdp_result
    assert "values" in pdp_result
    assert "average_prediction" in pdp_result
    assert pdp_result["feature"] == feature
    assert len(pdp_result["values"]) > 0
    assert len(pdp_result["values"]) == len(pdp_result["average_prediction"])
    
    # Test visualization
    fig = pd_plot.plot_partial_dependence(pdp_result)
    assert isinstance(fig, plt.Figure)


def test_pdp_classification(
    classification_model, sample_classification_data
):
    """Test partial dependence plots with classification tasks."""
    # Create PDP generator
    pd_plot = PDPlot(classification_model)
    
    # Get a feature to visualize
    feature = sample_classification_data["feature_names"][0]
    
    # Generate PDP
    pdp_result = pd_plot.compute_partial_dependence(
        data=sample_classification_data["test"],
        feature=feature,
        task_name="classification",
    )
    
    # Verify output structure and content
    assert pdp_result is not None
    assert "feature" in pdp_result
    assert "values" in pdp_result
    assert "average_prediction" in pdp_result
    assert pdp_result["feature"] == feature
    assert len(pdp_result["values"]) > 0
    assert len(pdp_result["values"]) == len(pdp_result["average_prediction"])
    
    # Test visualization
    fig = pd_plot.plot_partial_dependence(pdp_result)
    assert isinstance(fig, plt.Figure)


def test_pdp_survival(
    survival_model, sample_survival_data
):
    """Test partial dependence plots with survival tasks."""
    # Create PDP generator
    pd_plot = PDPlot(survival_model)
    
    # Get a feature to visualize
    feature = sample_survival_data["feature_names"][0]
    
    # Generate PDP
    pdp_result = pd_plot.compute_partial_dependence(
        data=sample_survival_data["test"],
        feature=feature,
        task_name="survival",
    )
    
    # Verify output structure and content
    assert pdp_result is not None
    assert "feature" in pdp_result
    assert "values" in pdp_result
    assert "average_prediction" in pdp_result
    assert pdp_result["feature"] == feature
    assert len(pdp_result["values"]) > 0
    assert len(pdp_result["values"]) == len(pdp_result["average_prediction"])
    
    # Test visualization
    fig = pd_plot.plot_partial_dependence(pdp_result)
    assert isinstance(fig, plt.Figure)


def test_pdp_count(
    count_model, sample_count_data
):
    """Test partial dependence plots with count regression tasks."""
    # Create PDP generator
    pd_plot = PDPlot(count_model)
    
    # Get a feature to visualize
    feature = sample_count_data["feature_names"][0]
    
    # Generate PDP
    pdp_result = pd_plot.compute_partial_dependence(
        data=sample_count_data["test"],
        feature=feature,
        task_name="count",
    )
    
    # Verify output structure and content
    assert pdp_result is not None
    assert "feature" in pdp_result
    assert "values" in pdp_result
    assert "average_prediction" in pdp_result
    assert pdp_result["feature"] == feature
    assert len(pdp_result["values"]) > 0
    assert len(pdp_result["values"]) == len(pdp_result["average_prediction"])
    
    # Test visualization
    fig = pd_plot.plot_partial_dependence(pdp_result)
    assert isinstance(fig, plt.Figure)


def test_pdp_competing_risks(
    competing_risks_model, sample_competing_risks_data
):
    """Test partial dependence plots with competing risks tasks."""
    # Create PDP generator
    pd_plot = PDPlot(competing_risks_model)
    
    # Get a feature to visualize
    feature = sample_competing_risks_data["feature_names"][0]
    
    # Generate PDP
    pdp_result = pd_plot.compute_partial_dependence(
        data=sample_competing_risks_data["test"],
        feature=feature,
        task_name="competing_risks",
    )
    
    # Verify output structure and content
    assert pdp_result is not None
    assert "feature" in pdp_result
    assert "values" in pdp_result
    assert "average_prediction" in pdp_result
    assert pdp_result["feature"] == feature
    assert len(pdp_result["values"]) > 0
    assert len(pdp_result["values"]) == len(pdp_result["average_prediction"])
    
    # Test visualization
    fig = pd_plot.plot_partial_dependence(pdp_result)
    assert isinstance(fig, plt.Figure)


def test_pdp_clustering(
    clustering_model, sample_clustering_data
):
    """Test partial dependence plots with clustering tasks."""
    # Create PDP generator
    pd_plot = PDPlot(clustering_model)
    
    # Get a feature to visualize
    feature = sample_clustering_data["feature_names"][0]
    
    # Generate PDP
    pdp_result = pd_plot.compute_partial_dependence(
        data=sample_clustering_data["test"],
        feature=feature,
        task_name="clustering",
    )
    
    # Verify output structure and content
    assert pdp_result is not None
    assert "feature" in pdp_result
    assert "values" in pdp_result
    assert "average_prediction" in pdp_result
    assert pdp_result["feature"] == feature
    assert len(pdp_result["values"]) > 0
    assert len(pdp_result["values"]) == len(pdp_result["average_prediction"])
    
    # Test visualization
    fig = pd_plot.plot_partial_dependence(pdp_result)
    assert isinstance(fig, plt.Figure)


def test_ice_regression(
    regression_model, sample_regression_data
):
    """Test ICE plots with regression tasks."""
    # Create ICE generator
    ice_plot = ICEPlot(regression_model)
    
    # Get a feature to visualize
    feature = sample_regression_data["feature_names"][0]
    
    # Generate ICE curves
    ice_result = ice_plot.compute_ice_curves(
        data=sample_regression_data["test"],
        feature=feature,
        task_name="regression",
        n_samples=5,  # Small number for testing
    )
    
    # Verify output structure and content
    assert ice_result is not None
    assert "feature" in ice_result
    assert "values" in ice_result
    assert "ice_curves" in ice_result
    assert "pd_curve" in ice_result
    assert "instances" in ice_result
    assert ice_result["feature"] == feature
    assert len(ice_result["values"]) > 0
    assert len(ice_result["ice_curves"]) == 5  # n_samples
    
    # Test visualization
    fig = ice_plot.plot_ice_curves(ice_result)
    assert isinstance(fig, plt.Figure)


def test_ice_classification(
    classification_model, sample_classification_data
):
    """Test ICE plots with classification tasks."""
    # Create ICE generator
    ice_plot = ICEPlot(classification_model)
    
    # Get a feature to visualize
    feature = sample_classification_data["feature_names"][0]
    
    # Generate ICE curves
    ice_result = ice_plot.compute_ice_curves(
        data=sample_classification_data["test"],
        feature=feature,
        task_name="classification",
        n_samples=5,  # Small number for testing
    )
    
    # Verify output structure and content
    assert ice_result is not None
    assert "feature" in ice_result
    assert "values" in ice_result
    assert "ice_curves" in ice_result
    assert "pd_curve" in ice_result
    assert "instances" in ice_result
    assert ice_result["feature"] == feature
    assert len(ice_result["values"]) > 0
    assert len(ice_result["ice_curves"]) == 5  # n_samples
    
    # Test visualization
    fig = ice_plot.plot_ice_curves(ice_result)
    assert isinstance(fig, plt.Figure)


def test_ice_survival(
    survival_model, sample_survival_data
):
    """Test ICE plots with survival tasks."""
    # Create ICE generator
    ice_plot = ICEPlot(survival_model)
    
    # Get a feature to visualize
    feature = sample_survival_data["feature_names"][0]
    
    # Generate ICE curves
    ice_result = ice_plot.compute_ice_curves(
        data=sample_survival_data["test"],
        feature=feature,
        task_name="survival",
        n_samples=5,  # Small number for testing
    )
    
    # Verify output structure and content
    assert ice_result is not None
    assert "feature" in ice_result
    assert "values" in ice_result
    assert "ice_curves" in ice_result
    assert "pd_curve" in ice_result
    assert "instances" in ice_result
    assert ice_result["feature"] == feature
    assert len(ice_result["values"]) > 0
    assert len(ice_result["ice_curves"]) == 5  # n_samples
    
    # Test visualization
    fig = ice_plot.plot_ice_curves(ice_result)
    assert isinstance(fig, plt.Figure)


def test_calibration_classification(
    classification_model, sample_classification_data
):
    """Test calibration plots with classification tasks."""
    # Create calibration plot generator
    cal_plot = CalibrationPlot(classification_model)
    
    # Generate calibration data
    cal_result = cal_plot.compute_calibration(
        data=sample_classification_data["test"],
        target_column="target",
        task_name="classification",
    )
    
    # Verify output structure and content
    assert cal_result is not None
    assert "prob_true" in cal_result
    assert "prob_pred" in cal_result
    assert "hist" in cal_result
    assert len(cal_result["prob_true"]) > 0
    assert len(cal_result["prob_pred"]) == len(cal_result["prob_true"])
    
    # Test visualization
    fig = cal_plot.plot_calibration(cal_result)
    assert isinstance(fig, plt.Figure)
