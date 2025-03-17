"""
Tests for report generation.

This module tests that report generation works with all task heads.
"""

import pytest
import os
import shutil
import numpy as np
import pandas as pd
from typing import Dict

from tabular_transformer.explainability.report import Report, DashboardBuilder


def test_report_regression(
    regression_model, sample_regression_data
):
    """Test report generation with regression tasks."""
    # Create report generator
    report_generator = Report(regression_model)
    
    # Create output path for test
    output_path = "test_regression_report.html"
    
    # Generate report
    report = report_generator.generate_report(
        data=sample_regression_data["test"],
        target_columns={"regression": "target"},
        task_names=["regression"],
        sample_instances=sample_regression_data["test"].iloc[:3],  # Use a few instances
        output_path=output_path,
    )
    
    # Verify the report was generated
    assert os.path.exists(output_path)
    
    # Verify report structure
    assert report is not None
    assert "global_explanations" in report
    assert "local_explanations" in report
    assert "visualizations" in report
    
    # Verify global explanations
    assert "feature_importance" in report["global_explanations"]
    
    # Verify local explanations
    assert len(report["local_explanations"]) > 0
    
    # Clean up
    os.remove(output_path)


def test_report_classification(
    classification_model, sample_classification_data
):
    """Test report generation with classification tasks."""
    # Create report generator
    report_generator = Report(classification_model)
    
    # Create output path for test
    output_path = "test_classification_report.html"
    
    # Generate report
    report = report_generator.generate_report(
        data=sample_classification_data["test"],
        target_columns={"classification": "target"},
        task_names=["classification"],
        sample_instances=sample_classification_data["test"].iloc[:3],  # Use a few instances
        output_path=output_path,
    )
    
    # Verify the report was generated
    assert os.path.exists(output_path)
    
    # Verify report structure
    assert report is not None
    assert "global_explanations" in report
    assert "local_explanations" in report
    assert "visualizations" in report
    
    # Verify global explanations
    assert "feature_importance" in report["global_explanations"]
    
    # Verify local explanations
    assert len(report["local_explanations"]) > 0
    
    # Clean up
    os.remove(output_path)


def test_report_survival(
    survival_model, sample_survival_data
):
    """Test report generation with survival tasks."""
    # Create report generator
    report_generator = Report(survival_model)
    
    # Create output path for test
    output_path = "test_survival_report.html"
    
    # Generate report
    report = report_generator.generate_report(
        data=sample_survival_data["test"],
        target_columns={
            "survival": [
                sample_survival_data["time_column"],
                sample_survival_data["event_column"]
            ]
        },
        task_names=["survival"],
        sample_instances=sample_survival_data["test"].iloc[:3][sample_survival_data["feature_names"]],
        output_path=output_path,
    )
    
    # Verify the report was generated
    assert os.path.exists(output_path)
    
    # Verify report structure
    assert report is not None
    assert "global_explanations" in report
    assert "local_explanations" in report
    assert "visualizations" in report
    
    # Verify global explanations
    assert "feature_importance" in report["global_explanations"]
    
    # Verify local explanations
    assert len(report["local_explanations"]) > 0
    
    # Clean up
    os.remove(output_path)


def test_report_count(
    count_model, sample_count_data
):
    """Test report generation with count regression tasks."""
    # Create report generator
    report_generator = Report(count_model)
    
    # Create output path for test
    output_path = "test_count_report.html"
    
    # Generate report
    report = report_generator.generate_report(
        data=sample_count_data["test"],
        target_columns={"count": "count"},
        task_names=["count"],
        sample_instances=sample_count_data["test"].iloc[:3][sample_count_data["feature_names"]],
        output_path=output_path,
    )
    
    # Verify the report was generated
    assert os.path.exists(output_path)
    
    # Verify report structure
    assert report is not None
    assert "global_explanations" in report
    assert "local_explanations" in report
    assert "visualizations" in report
    
    # Verify global explanations
    assert "feature_importance" in report["global_explanations"]
    
    # Verify local explanations
    assert len(report["local_explanations"]) > 0
    
    # Clean up
    os.remove(output_path)


def test_report_competing_risks(
    competing_risks_model, sample_competing_risks_data
):
    """Test report generation with competing risks tasks."""
    # Create report generator
    report_generator = Report(competing_risks_model)
    
    # Create output path for test
    output_path = "test_competing_risks_report.html"
    
    # Generate report
    report = report_generator.generate_report(
        data=sample_competing_risks_data["test"],
        target_columns={
            "competing_risks": [
                sample_competing_risks_data["time_column"],
                sample_competing_risks_data["event_column"]
            ]
        },
        task_names=["competing_risks"],
        sample_instances=sample_competing_risks_data["test"].iloc[:3][sample_competing_risks_data["feature_names"]],
        output_path=output_path,
    )
    
    # Verify the report was generated
    assert os.path.exists(output_path)
    
    # Verify report structure
    assert report is not None
    assert "global_explanations" in report
    assert "local_explanations" in report
    assert "visualizations" in report
    
    # Verify global explanations
    assert "feature_importance" in report["global_explanations"]
    
    # Verify local explanations
    assert len(report["local_explanations"]) > 0
    
    # Clean up
    os.remove(output_path)


def test_report_clustering(
    clustering_model, sample_clustering_data
):
    """Test report generation with clustering tasks."""
    # Create report generator
    report_generator = Report(clustering_model)
    
    # Create output path for test
    output_path = "test_clustering_report.html"
    
    # Generate report
    report = report_generator.generate_report(
        data=sample_clustering_data["test"],
        target_columns={},  # No target columns for clustering
        task_names=["clustering"],
        sample_instances=sample_clustering_data["test"].iloc[:3],
        output_path=output_path,
    )
    
    # Verify the report was generated
    assert os.path.exists(output_path)
    
    # Verify report structure
    assert report is not None
    assert "global_explanations" in report
    assert "local_explanations" in report
    assert "visualizations" in report
    
    # Verify global explanations
    assert "feature_importance" in report["global_explanations"]
    
    # Verify local explanations
    assert len(report["local_explanations"]) > 0
    
    # Clean up
    os.remove(output_path)


def test_dashboard_regression(
    regression_model, sample_regression_data
):
    """Test dashboard generation with regression tasks."""
    # Create report generator for dashboard builder
    report_generator = Report(regression_model)
    
    # Create dashboard builder
    dashboard_builder = DashboardBuilder(report_generator)
    
    # Create output directory for test
    output_directory = "test_regression_dashboard"
    
    # Build dashboard
    dashboard_path = dashboard_builder.build_dashboard(
        data=sample_regression_data["test"],
        target_columns={"regression": "target"},
        output_directory=output_directory,
    )
    
    # Verify the dashboard was generated
    assert os.path.exists(dashboard_path)
    assert os.path.isfile(dashboard_path)
    
    # Clean up
    shutil.rmtree(output_directory)


def test_dashboard_classification(
    classification_model, sample_classification_data
):
    """Test dashboard generation with classification tasks."""
    # Create report generator for dashboard builder
    report_generator = Report(classification_model)
    
    # Create dashboard builder
    dashboard_builder = DashboardBuilder(report_generator)
    
    # Create output directory for test
    output_directory = "test_classification_dashboard"
    
    # Build dashboard
    dashboard_path = dashboard_builder.build_dashboard(
        data=sample_classification_data["test"],
        target_columns={"classification": "target"},
        output_directory=output_directory,
    )
    
    # Verify the dashboard was generated
    assert os.path.exists(dashboard_path)
    assert os.path.isfile(dashboard_path)
    
    # Clean up
    shutil.rmtree(output_directory)
