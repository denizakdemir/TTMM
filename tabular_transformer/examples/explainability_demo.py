"""
Explainability demo for tabular transformer.

This script demonstrates how to use the explainability components
to explain and visualize a trained tabular transformer model.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabular_transformer.utils.config import TransformerConfig, ModelConfig
from tabular_transformer.models.transformer_encoder import TabularTransformer
from tabular_transformer.models.task_heads.regression import RegressionHead
from tabular_transformer.data.preprocess import FeaturePreprocessor
from tabular_transformer.data.dataset import TabularDataset
from tabular_transformer.training.trainer import Trainer
from tabular_transformer.inference.predict import Predictor

# Import explainability components
from tabular_transformer.explainability.global_explanations import (
    GlobalExplainer, PermutationImportance
)
from tabular_transformer.explainability.local_explanations import (
    LIMEExplainer, CounterfactualExplainer
)
from tabular_transformer.explainability.visualizations import (
    PDPlot, ICEPlot, CalibrationPlot
)
from tabular_transformer.explainability.sensitivity import SensitivityAnalyzer
from tabular_transformer.explainability.report import Report, DashboardBuilder


def create_demo_model_and_data():
    """
    Create a simple demo model and data for explainability demonstration.
    
    Returns:
        Tuple of (predictor, train_data, test_data)
    """
    print("Creating demo model and data...")
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    feature_names = diabetes.feature_names
    X = pd.DataFrame(diabetes.data, columns=feature_names)
    y = pd.Series(diabetes.target, name="target")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create training and test dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Create preprocessor
    preprocessor = FeaturePreprocessor(
        numeric_columns=feature_names,
        categorical_columns=[],
    )
    preprocessor.fit(train_df)
    
    # Create dataset
    train_dataset = TabularDataset(
        dataframe=train_df,
        numeric_columns=feature_names,
        categorical_columns=[],
        target_columns={"regression": ["target"]},
        preprocessor=preprocessor,
        fit_preprocessor=False,
    )
    
    # Create model
    config = TransformerConfig(
        input_dim=len(feature_names),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        variational=False,
    )
    
    encoder = TabularTransformer(
        numeric_dim=len(feature_names),
        categorical_dims={},
        categorical_embedding_dims={},
        config=config,
    )
    
    # Create task head
    task_head = RegressionHead(
        name="regression",
        input_dim=32,  # Matches encoder embed_dim
        hidden_dims=[16],
        dropout=0.1,
    )
    
    # Create trainer (without ModelConfig)
    trainer = Trainer(
        encoder=encoder,
        task_head={"regression": task_head},  # Changed from task_heads to task_head
        optimizer=torch.optim.Adam(
            list(encoder.parameters()) + list(task_head.parameters()),
            lr=0.001,
            weight_decay=0.0001
        ),
        device="cpu",
    )
    
    # Create data loader
    train_loader = train_dataset.create_dataloader(batch_size=32)
    
    # Train model (lightly)
    trainer.train(
        train_loader=train_loader,
        val_loader=None,
        num_epochs=5,
    )
    
    # Create predictor
    predictor = Predictor(
        encoder=encoder,
        task_head={"regression": task_head},  # Changed from task_heads to task_head
        preprocessor=preprocessor,
        device="cpu",
    )
    
    print("Model training complete.")
    return predictor, train_df, test_df


def demonstrate_global_explanations(predictor, train_df, test_df):
    """
    Demonstrate global explanation methods.
    
    Args:
        predictor: Trained predictor
        train_df: Training data
        test_df: Test data
    """
    print("\n=== Global Explanations ===")
    
    # Create global explainer
    global_explainer = GlobalExplainer(predictor)
    
    # Create permutation importance calculator
    perm_importance = PermutationImportance(predictor)
    
    # Compute feature importance
    target_columns = {"regression": "target"}
    importance = perm_importance.compute_importance(
        data=test_df,
        target_columns=target_columns,
        task_names=["regression"],
    )
    
    # Plot feature importance
    fig = global_explainer.plot_feature_importance(importance)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close(fig)
    
    print("Feature importance plot saved to 'feature_importance.png'")


def demonstrate_local_explanations(predictor, test_df):
    """
    Demonstrate local explanation methods.
    
    Args:
        predictor: Trained predictor
        test_df: Test data
    """
    print("\n=== Local Explanations ===")
    
    # Create LIME explainer
    lime_explainer = LIMEExplainer(predictor)
    
    # Create counterfactual explainer
    cf_explainer = CounterfactualExplainer(predictor)
    
    # Get a sample instance
    instance = test_df.iloc[0]
    
    # Generate LIME explanation
    lime_explanation = lime_explainer.explain_instance(
        instance=instance,
        task_name="regression",
    )
    
    # Plot LIME explanation
    lime_fig = lime_explainer.plot_explanation(lime_explanation)
    plt.tight_layout()
    plt.savefig("lime_explanation.png")
    plt.close(lime_fig)
    
    print("LIME explanation plot saved to 'lime_explanation.png'")
    
    # Generate counterfactual explanation
    cf_explanation = cf_explainer.explain_instance(
        instance=instance,
        task_name="regression",
    )
    
    # Plot counterfactual explanation
    cf_fig = cf_explainer.plot_explanation(cf_explanation)
    plt.tight_layout()
    plt.savefig("counterfactual_explanation.png")
    plt.close(cf_fig)
    
    print("Counterfactual explanation plot saved to 'counterfactual_explanation.png'")


def demonstrate_visualizations(predictor, train_df, test_df):
    """
    Demonstrate visualization methods.
    
    Args:
        predictor: Trained predictor
        train_df: Training data
        test_df: Test data
    """
    print("\n=== Visualization Plots ===")
    
    # Create PDP and ICE plot generators
    pd_plot = PDPlot(predictor)
    ice_plot = ICEPlot(predictor)
    
    # Get feature names
    feature_names = predictor.preprocessor.numeric_columns
    
    # Choose a feature to visualize
    feature = feature_names[0]  # First feature
    
    # Generate PDP
    pdp_result = pd_plot.compute_partial_dependence(
        data=test_df,
        feature=feature,
        task_name="regression",
    )
    
    # Plot PDP
    pdp_fig = pd_plot.plot_partial_dependence(pdp_result)
    plt.tight_layout()
    plt.savefig("pdp_plot.png")
    plt.close(pdp_fig)
    
    print(f"PDP plot for feature '{feature}' saved to 'pdp_plot.png'")
    
    # Generate ICE curves
    ice_result = ice_plot.compute_ice_curves(
        data=test_df,
        feature=feature,
        task_name="regression",
        n_samples=10,  # Use 10 samples for clarity
    )
    
    # Plot ICE curves
    ice_fig = ice_plot.plot_ice_curves(ice_result)
    plt.tight_layout()
    plt.savefig("ice_plot.png")
    plt.close(ice_fig)
    
    print(f"ICE plot for feature '{feature}' saved to 'ice_plot.png'")


def demonstrate_sensitivity(predictor, test_df):
    """
    Demonstrate sensitivity analysis.
    
    Args:
        predictor: Trained predictor
        test_df: Test data
    """
    print("\n=== Sensitivity Analysis ===")
    
    # Create sensitivity analyzer
    sensitivity_analyzer = SensitivityAnalyzer(predictor)
    
    # Get a sample instance
    instance = test_df.iloc[0]
    
    # Analyze sensitivity
    sensitivity_result = sensitivity_analyzer.compute_sensitivity(
        instance=instance,
        task_name="regression",
        n_samples=50,  # Use 50 samples for speed
    )
    
    # Plot sensitivity
    sensitivity_fig = sensitivity_analyzer.plot_sensitivity(sensitivity_result)
    plt.tight_layout()
    plt.savefig("sensitivity_plot.png")
    plt.close(sensitivity_fig)
    
    print("Sensitivity plot saved to 'sensitivity_plot.png'")
    
    # Plot tornado diagram
    tornado_fig = sensitivity_analyzer.plot_tornado(sensitivity_result)
    plt.tight_layout()
    plt.savefig("tornado_plot.png")
    plt.close(tornado_fig)
    
    print("Tornado plot saved to 'tornado_plot.png'")


def demonstrate_report(predictor, train_df, test_df):
    """
    Demonstrate report generation.
    
    Args:
        predictor: Trained predictor
        train_df: Training data
        test_df: Test data
    """
    print("\n=== Report Generation ===")
    
    # Create report generator
    report_generator = Report(predictor)
    
    # Generate report
    report = report_generator.generate_report(
        data=test_df,
        target_columns={"regression": "target"},
        task_names=["regression"],
        sample_instances=test_df.iloc[:3],  # Use first 3 instances
        output_path="explainability_report.html",
    )
    
    print("Explainability report saved to 'explainability_report.html'")
    
    # Create dashboard builder
    dashboard_builder = DashboardBuilder(report_generator)
    
    # Create output directory
    os.makedirs("explainability_dashboard", exist_ok=True)
    
    # Build dashboard
    dashboard_path = dashboard_builder.build_dashboard(
        data=test_df,
        target_columns={"regression": "target"},
        output_directory="explainability_dashboard",
    )
    
    print(f"Explainability dashboard saved to '{dashboard_path}'")


def main():
    """
    Run the explainability demo.
    """
    print("=== Tabular Transformer Explainability Demo ===\n")
    
    # Create model and data
    predictor, train_df, test_df = create_demo_model_and_data()
    
    # Demonstrate global explanations
    demonstrate_global_explanations(predictor, train_df, test_df)
    
    # Demonstrate local explanations
    demonstrate_local_explanations(predictor, test_df)
    
    # Demonstrate visualizations
    demonstrate_visualizations(predictor, train_df, test_df)
    
    # Demonstrate sensitivity analysis
    demonstrate_sensitivity(predictor, test_df)
    
    # Demonstrate report generation
    demonstrate_report(predictor, train_df, test_df)
    
    print("\nDemo complete. Generated files:")
    print("- feature_importance.png")
    print("- lime_explanation.png")
    print("- counterfactual_explanation.png")
    print("- pdp_plot.png")
    print("- ice_plot.png")
    print("- sensitivity_plot.png")
    print("- tornado_plot.png")
    print("- explainability_report.html")
    print("- explainability_dashboard/dashboard.html")


if __name__ == "__main__":
    main()
