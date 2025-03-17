# Tabular Transformer Explainability Module

This module provides comprehensive explainability components for the Tabular Transformer model. It enables users to understand and visualize model behavior at both global and local levels.

## Components

### Global Explainability

- **`GlobalExplainer`**: Base class for global model explanations
- **`PermutationImportance`**: Measures feature importance by permuting feature values
- **`IntegratedGradients`**: Computes importance by integrating gradients along a path
- **`SHAPExplainer`**: Calculates SHAP (SHapley Additive exPlanations) values
- **`AttentionExplainer`**: Extracts and analyzes attention weights from the transformer

### Local Explainability

- **`LocalExplainer`**: Base class for local instance-level explanations
- **`LIMEExplainer`**: Approximates the model locally with an interpretable model
- **`CounterfactualExplainer`**: Generates counterfactual examples showing minimal changes needed to alter predictions

### Visualizations

- **`ExplainabilityViz`**: Base class for visualization utilities
- **`PDPlot`**: Creates Partial Dependence Plots showing marginal effect of features
- **`ICEPlot`**: Creates Individual Conditional Expectation plots showing how predictions change for individual instances
- **`CalibrationPlot`**: Generates calibration plots comparing predicted probabilities to observed outcomes

### Sensitivity Analysis

- **`SensitivityAnalyzer`**: Analyzes model sensitivity to feature variations for robustness evaluation

### Reporting

- **`Report`**: Generates comprehensive reports combining multiple explainability methods
- **`DashboardBuilder`**: Creates interactive dashboards for exploring model behavior

## Usage

```python
from tabular_transformer.explainability import (
    GlobalExplainer, PermutationImportance,
    LIMEExplainer, CounterfactualExplainer,
    PDPlot, ICEPlot, CalibrationPlot,
    SensitivityAnalyzer, Report
)

# Initialize with a trained predictor
global_explainer = GlobalExplainer(predictor)
perm_importance = PermutationImportance(predictor)

# Compute feature importance
importance = perm_importance.compute_importance(
    data=test_data,
    target_columns={"regression": "target"},
    task_names=["regression"],
)

# Plot feature importance
fig = global_explainer.plot_feature_importance(importance)

# Generate comprehensive report
report_generator = Report(predictor)
report = report_generator.generate_report(
    data=test_data,
    target_columns={"regression": "target"},
    task_names=["regression"],
    sample_instances=test_data.iloc[:5],
    output_path="explainability_report.html",
)
```

See the `explainability_demo.py` example for a complete demonstration of all components.

## Installation

To use the explainability module, install the required dependencies:

```
pip install tabular_transformer[explainability]
```

For dashboard functionality:

```
pip install tabular_transformer[dashboard]
```

## References

The module implements methods based on the following key papers and frameworks:

- [Ribeiro et al. "Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) (LIME)
- [Lundberg & Lee. A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) (SHAP)
- [Sundararajan et al. Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) (Integrated Gradients)
- [Friedman. Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) (Partial Dependence Plots)
- [Goldstein et al. Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation](https://arxiv.org/abs/1309.6392) (ICE Plots)
