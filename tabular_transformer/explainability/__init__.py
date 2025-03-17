"""
Explainability module for tabular transformer.

This module provides components for explaining and visualizing
model predictions and behavior at both global and local levels.
"""

from tabular_transformer.explainability.global_explanations import (
    GlobalExplainer, 
    PermutationImportance,
    IntegratedGradients,
    SHAPExplainer,
    AttentionExplainer
)
from tabular_transformer.explainability.local_explanations import (
    LocalExplainer,
    LIMEExplainer,
    CounterfactualExplainer
)
from tabular_transformer.explainability.visualizations import (
    ExplainabilityViz,
    PDPlot,
    ICEPlot,
    CalibrationPlot
)
from tabular_transformer.explainability.sensitivity import (
    SensitivityAnalyzer
)
from tabular_transformer.explainability.report import (
    Report,
    DashboardBuilder
)

__all__ = [
    # Global explainability
    'GlobalExplainer', 'PermutationImportance', 'IntegratedGradients',
    'SHAPExplainer', 'AttentionExplainer',
    # Local explainability
    'LocalExplainer', 'LIMEExplainer', 'CounterfactualExplainer',
    # Visualizations
    'ExplainabilityViz', 'PDPlot', 'ICEPlot', 'CalibrationPlot',
    # Sensitivity analysis
    'SensitivityAnalyzer',
    # Reporting
    'Report', 'DashboardBuilder',
]
