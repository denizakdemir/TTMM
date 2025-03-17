"""
Sensitivity analysis utilities for tabular transformer.

This module provides classes and functions for analyzing model sensitivity
to input feature variations and robustness to perturbations.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from tabular_transformer.utils.logger import LoggerMixin
from tabular_transformer.inference.predict import Predictor
from tabular_transformer.data.dataset import TabularDataset


class SensitivityAnalyzer(LoggerMixin):
    """
    Sensitivity analyzer for tabular transformer.
    
    This class provides methods to assess how variations in input features
    impact model predictions, helping understand model robustness.
    """
    
    def __init__(
        self,
        predictor: Predictor,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize sensitivity analyzer.
        
        Args:
            predictor: Predictor instance with trained model
            feature_names: Optional list of feature names (will use preprocessor column names if not provided)
        """
        self.predictor = predictor
        self.encoder = predictor.encoder
        self.task_heads = predictor.task_heads
        self.preprocessor = predictor.preprocessor
        self.device = predictor.device
        
        # Set feature names
        if feature_names is None:
            numeric_cols = self.preprocessor.numeric_columns
            categorical_cols = self.preprocessor.categorical_columns
            self.feature_names = numeric_cols + categorical_cols
        else:
            self.feature_names = feature_names
    
    def compute_sensitivity(
        self,
        instance: Union[pd.Series, pd.DataFrame],
        task_name: str,
        features_to_analyze: Optional[List[str]] = None,
        perturbation_scales: Optional[Dict[str, float]] = None,
        n_samples: int = 100,
        batch_size: int = 64,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute sensitivity of model predictions to feature perturbations.
        
        Args:
            instance: Input instance (Series or single-row DataFrame)
            task_name: Name of the task to analyze
            features_to_analyze: List of features to analyze (all numeric features by default)
            perturbation_scales: Dict mapping features to perturbation scales (fraction of std)
            n_samples: Number of perturbation samples per feature
            batch_size: Batch size for prediction
            random_state: Random seed
            
        Returns:
            Dict containing sensitivity analysis results
        """
        # Convert instance to DataFrame if needed
        if isinstance(instance, pd.Series):
            instance_df = pd.DataFrame([instance])
        else:
            instance_df = instance
            
        # Check if task is valid
        if task_name not in self.task_heads:
            raise ValueError(f"Task '{task_name}' not found in model")
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
        
        # Determine which features to analyze
        if features_to_analyze is None:
            # Use numeric features by default
            features_to_analyze = self.preprocessor.numeric_columns
        else:
            # Filter to include only numeric features
            features_to_analyze = [
                f for f in features_to_analyze if f in self.preprocessor.numeric_columns
            ]
            
        if not features_to_analyze:
            raise ValueError("No valid numeric features to analyze")
        
        # Set default perturbation scales if not provided
        if perturbation_scales is None:
            perturbation_scales = {f: 0.1 for f in features_to_analyze}
        
        # Get original prediction
        original_pred = self.predictor.predict_dataframe(
            df=instance_df,
            task_names=[task_name],
            batch_size=1,
        )
        
        # Extract original prediction value
        task_pred = original_pred[task_name]
        
        if "prediction" in task_pred.columns:
            # Regression
            original_value = task_pred["prediction"].values[0]
        elif "class_probs_1" in task_pred.columns:
            # Binary classification
            original_value = task_pred["class_probs_1"].values[0]
        else:
            # Default to first column
            first_col = task_pred.columns[0]
            original_value = task_pred[first_col].values[0]
        
        # Store sensitivity results
        sensitivity_results = {}
        
        # Analyze each feature
        for feature in features_to_analyze:
            # Create perturbed samples
            perturbed_dfs = []
            perturbation_values = []
            
            # Get original value
            original_feature_value = instance_df[feature].values[0]
            
            # Determine perturbation scale
            scale = perturbation_scales.get(feature, 0.1)
            
            # Use standard deviation of feature from preprocessor if available
            if hasattr(self.preprocessor, "numeric_scaler"):
                if hasattr(self.preprocessor.numeric_scaler, "scale_"):
                    feature_idx = self.preprocessor.numeric_columns.index(feature)
                    std = self.preprocessor.numeric_scaler.scale_[feature_idx]
                    perturbation_range = std * scale
                else:
                    # Fallback to 10% of the original value
                    perturbation_range = abs(original_feature_value) * scale
            else:
                # Fallback to 10% of the original value
                perturbation_range = abs(original_feature_value) * scale
            
            # Generate perturbed values
            for _ in range(n_samples):
                # Create a perturbed sample
                perturbed_df = instance_df.copy()
                
                # Apply perturbation
                perturbation = np.random.normal(0, perturbation_range)
                perturbed_value = original_feature_value + perturbation
                perturbed_df[feature] = perturbed_value
                
                # Store
                perturbed_dfs.append(perturbed_df)
                perturbation_values.append(perturbation)
            
            # Combine perturbed samples
            combined_df = pd.concat(perturbed_dfs, axis=0)
            
            # Get predictions for perturbed samples
            perturbed_preds = self.predictor.predict_dataframe(
                df=combined_df,
                task_names=[task_name],
                batch_size=batch_size,
            )
            
            # Extract prediction values
            task_pred = perturbed_preds[task_name]
            
            if "prediction" in task_pred.columns:
                # Regression
                pred_values = task_pred["prediction"].values
            elif "class_probs_1" in task_pred.columns:
                # Binary classification
                pred_values = task_pred["class_probs_1"].values
            else:
                # Default to first column
                first_col = task_pred.columns[0]
                pred_values = task_pred[first_col].values
            
            # Calculate prediction changes
            pred_changes = pred_values - original_value
            
            # Store results
            sensitivity_results[feature] = {
                "perturbations": np.array(perturbation_values),
                "predictions": pred_values,
                "pred_changes": pred_changes,
                "original_value": original_feature_value,
                "original_prediction": original_value,
            }
            
            # Calculate sensitivity metrics
            sensitivity_results[feature]["correlation"] = np.corrcoef(
                perturbation_values, pred_changes
            )[0, 1]
            
            sensitivity_results[feature]["sensitivity"] = np.mean(
                np.abs(pred_changes / np.array(perturbation_values))
            ) if not np.all(np.array(perturbation_values) == 0) else 0
        
        return {
            "task": task_name,
            "instance": instance,
            "original_prediction": original_value,
            "feature_results": sensitivity_results,
        }
    
    def plot_sensitivity(
        self,
        sensitivity_result: Dict[str, Any],
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Plot sensitivity analysis results.
        
        Args:
            sensitivity_result: Results from compute_sensitivity
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Get feature results
        feature_results = sensitivity_result["feature_results"]
        features = list(feature_results.keys())
        
        # Create figure with multiple subplots
        n_features = len(features)
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=figsize,
            squeeze=False,
        )
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot scatter for each feature
        for i, feature in enumerate(features):
            ax = axes[i]
            result = feature_results[feature]
            
            # Create scatter plot
            ax.scatter(
                result["perturbations"],
                result["pred_changes"],
                alpha=0.6,
            )
            
            # Add trend line
            x = result["perturbations"]
            y = result["pred_changes"]
            coef = np.polyfit(x, y, 1)
            poly1d_fn = np.poly1d(coef)
            ax.plot(
                np.sort(x),
                poly1d_fn(np.sort(x)),
                color="red",
                linestyle="--",
            )
            
            # Add correlation info
            corr = result["correlation"]
            sens = result["sensitivity"]
            ax.annotate(
                f"Correlation: {corr:.2f}\nSensitivity: {sens:.3f}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                verticalalignment="top",
                fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.1),
            )
            
            # Set labels
            ax.set_xlabel(f"Perturbation to {feature}")
            ax.set_ylabel("Change in Prediction")
            ax.set_title(f"Sensitivity to {feature}")
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            
            # Add vertical line at x=0
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_tornado(
        self,
        sensitivity_result: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 8),
        sort_by: str = "sensitivity",
    ) -> plt.Figure:
        """
        Plot tornado diagram of feature sensitivities.
        
        Args:
            sensitivity_result: Results from compute_sensitivity
            figsize: Figure size
            sort_by: Sort features by "sensitivity" or "correlation"
            
        Returns:
            Matplotlib figure
        """
        # Get feature results
        feature_results = sensitivity_result["feature_results"]
        features = list(feature_results.keys())
        
        # Extract sensitivity or correlation values
        if sort_by == "sensitivity":
            values = [feature_results[f]["sensitivity"] for f in features]
            metric_name = "Sensitivity"
        else:
            values = [abs(feature_results[f]["correlation"]) for f in features]
            metric_name = "Correlation (absolute)"
        
        # Create DataFrame for sorting
        df = pd.DataFrame({
            "Feature": features,
            metric_name: values,
        })
        
        # Sort by value
        df = df.sort_values(metric_name)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        ax.barh(df["Feature"], df[metric_name])
        
        # Set labels
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Feature")
        ax.set_title(f"Tornado Diagram: Feature {metric_name}")
        
        plt.tight_layout()
        return fig
