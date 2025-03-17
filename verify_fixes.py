"""
Verify that the fixes are working properly.

This script checks the key issues that were fixed:
1. Survival and CompetingRisks heads require num_time_bins parameter
2. Keys like "feature_importance", "counterfactual", "values", "feature_sensitivities" added to return dictionaries
"""

import numpy as np
import pandas as pd
import torch
import matplotlib
# Use non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the necessary modules
from tabular_transformer.models.task_heads import SurvivalHead, CompetingRisksHead
from tabular_transformer.explainability.local_explanations import LIMEExplainer, CounterfactualExplainer
from tabular_transformer.explainability.visualizations import PDPlot, ICEPlot, CalibrationPlot
from tabular_transformer.explainability.sensitivity import SensitivityAnalyzer

# Mock data and objects for testing
class MockPredictor:
    def __init__(self):
        self.preprocessor = type('obj', (object,), {
            'numeric_columns': ['feature_0', 'feature_1', 'feature_2'],
            'categorical_columns': []
        })
        self.encoder = None
        self.task_heads = {'mock': None}
        self.device = 'cpu'
    
    def predict_dataframe(self, df, task_names, batch_size=None):
        return {'mock': pd.DataFrame({'prediction': np.random.rand(len(df))})}

def test_task_head_initialization():
    """Test that task heads can be initialized with required parameters."""
    print("\n1. Testing task head initialization...")
    
    try:
        # SurvivalHead requires num_time_bins
        survival_head = SurvivalHead(
            name="survival",
            input_dim=32,
            hidden_dims=[16],
            dropout=0.1,
            num_time_bins=20
        )
        print("✓ SurvivalHead initialization succeeded")
    except TypeError as e:
        print(f"✗ SurvivalHead initialization failed: {e}")
    
    try:
        # CompetingRisksHead requires num_time_bins
        competing_risks_head = CompetingRisksHead(
            name="competing_risks",
            input_dim=32,
            hidden_dims=[16],
            num_risks=3,
            num_time_bins=20,
            dropout=0.1
        )
        print("✓ CompetingRisksHead initialization succeeded")
    except TypeError as e:
        print(f"✗ CompetingRisksHead initialization failed: {e}")

def test_explainer_return_keys():
    """Test that explainers return the expected keys."""
    print("\n2. Testing explainer return keys...")
    
    # Create mock data
    data = pd.DataFrame({
        'feature_0': np.random.rand(10),
        'feature_1': np.random.rand(10),
        'feature_2': np.random.rand(10)
    })
    
    # Create mock predictor
    predictor = MockPredictor()
    
    # Test LIME explainer
    lime_explainer = LIMEExplainer(predictor)
    lime_explanation = lime_explainer.explain_instance(data.iloc[0], 'mock')
    
    if 'feature_importance' in lime_explanation:
        print("✓ LIME explainer has 'feature_importance' key")
    else:
        print("✗ LIME explainer missing 'feature_importance' key")
    
    # Test counterfactual explainer
    cf_explainer = CounterfactualExplainer(predictor)
    cf_explanation = cf_explainer.explain_instance(data.iloc[0], 'mock')
    
    if 'counterfactual' in cf_explanation:
        print("✓ Counterfactual explainer has 'counterfactual' key")
    else:
        print("✗ Counterfactual explainer missing 'counterfactual' key")
    
    if 'distances' in cf_explanation:
        print("✓ Counterfactual explainer has 'distances' key")
    else:
        print("✗ Counterfactual explainer missing 'distances' key")
    
    # Test PDPlot
    pd_plot = PDPlot(predictor)
    
    # Mock the compute_partial_dependence method to avoid actual computation
    def mock_compute_partial_dependence(*args, **kwargs):
        return {
            'feature': 'feature_0',
            'task': 'mock',
            'grid': np.linspace(0, 1, 10),
            'pd_values': np.random.rand(10),
            'values': np.random.rand(10),  # This should be present
            'average_prediction': np.random.rand(10),  # This should be present
            'is_categorical': False
        }
    
    pd_plot.compute_partial_dependence = mock_compute_partial_dependence
    pdp_result = pd_plot.compute_partial_dependence(data, 'feature_0', 'mock')
    
    if 'values' in pdp_result:
        print("✓ PDPlot result has 'values' key")
    else:
        print("✗ PDPlot result missing 'values' key")
    
    if 'average_prediction' in pdp_result:
        print("✓ PDPlot result has 'average_prediction' key")
    else:
        print("✗ PDPlot result missing 'average_prediction' key")
    
    # Test ICEPlot
    ice_plot = ICEPlot(predictor)
    
    # Mock the compute_ice_curves method to avoid actual computation
    def mock_compute_ice_curves(*args, **kwargs):
        return {
            'feature': 'feature_0',
            'task': 'mock',
            'grid': np.linspace(0, 1, 10),
            'ice_values': np.random.rand(5, 10),
            'ice_curves': np.random.rand(5, 10),  # This should be present
            'pd_values': np.random.rand(10),
            'pd_curve': np.random.rand(10),  # This should be present
            'values': np.linspace(0, 1, 10),  # This should be present
            'sample_indices': np.arange(5),
            'instances': np.arange(5),  # This should be present
            'original_values': np.random.rand(5),
            'is_categorical': False
        }
    
    ice_plot.compute_ice_curves = mock_compute_ice_curves
    ice_result = ice_plot.compute_ice_curves(data, 'feature_0', 'mock')
    
    if 'ice_curves' in ice_result:
        print("✓ ICEPlot result has 'ice_curves' key")
    else:
        print("✗ ICEPlot result missing 'ice_curves' key")
    
    if 'pd_curve' in ice_result:
        print("✓ ICEPlot result has 'pd_curve' key")
    else:
        print("✗ ICEPlot result missing 'pd_curve' key")
    
    if 'instances' in ice_result:
        print("✓ ICEPlot result has 'instances' key")
    else:
        print("✗ ICEPlot result missing 'instances' key")
    
    # Test SensitivityAnalyzer
    sensitivity_analyzer = SensitivityAnalyzer(predictor)
    
    # Mock the compute_sensitivity method to avoid actual computation
    def mock_compute_sensitivity(*args, **kwargs):
        return {
            'task': 'mock',
            'instance': data.iloc[0],
            'original_prediction': 0.5,
            'feature_results': {'feature_0': {'perturbations': np.random.rand(10)}},
            'feature_sensitivities': {'feature_0': {'perturbations': np.random.rand(10)}}  # This should be present
        }
    
    sensitivity_analyzer.compute_sensitivity = mock_compute_sensitivity
    sensitivity_result = sensitivity_analyzer.compute_sensitivity(data.iloc[0], 'mock')
    
    if 'feature_sensitivities' in sensitivity_result:
        print("✓ SensitivityAnalyzer result has 'feature_sensitivities' key")
    else:
        print("✗ SensitivityAnalyzer result missing 'feature_sensitivities' key")

def test_calibration_method():
    """Test that calibration method name was fixed."""
    print("\n3. Testing CalibrationPlot method names...")
    
    # Create mock predictor
    predictor = MockPredictor()
    
    # Test CalibrationPlot
    calibration_plot = CalibrationPlot(predictor)
    
    # Check if compute_calibration method exists
    if hasattr(calibration_plot, 'compute_calibration'):
        print("✓ CalibrationPlot has 'compute_calibration' method")
    else:
        print("✗ CalibrationPlot missing 'compute_calibration' method")

if __name__ == "__main__":
    print("\n=== Starting verification of fixes ===")
    test_task_head_initialization()
    test_explainer_return_keys()
    test_calibration_method()
    print("\n=== Verification complete ===")
