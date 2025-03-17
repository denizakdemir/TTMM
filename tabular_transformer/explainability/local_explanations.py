"""
Local explainability methods for tabular transformer.

This module provides classes and functions for explaining individual predictions
at a local level, helping understand why specific predictions were made.
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


class LocalExplainer(LoggerMixin):
    """
    Base class for local explainability methods.
    
    This class provides common functionality for explaining individual
    predictions, focusing on feature contributions for specific instances.
    """
    
    def __init__(
        self,
        predictor: Predictor,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize local explainer.
        
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
    
    def explain_instance(
        self,
        instance: Union[pd.Series, pd.DataFrame],
        task_name: str,
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single instance.
        
        Args:
            instance: Input instance to explain (Series or single-row DataFrame)
            task_name: Name of the task to explain
            
        Returns:
            Dict containing explanation components
        """
        raise NotImplementedError("Subclasses must implement explain_instance")
    
    def plot_explanation(
        self,
        explanation: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot explanation for a single instance.
        
        Args:
            explanation: Explanation dict from explain_instance
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        raise NotImplementedError("Subclasses must implement plot_explanation")


class LIMEExplainer(LocalExplainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations) for tabular transformer.
    
    This class uses LIME to explain individual predictions by approximating
    the model locally with an interpretable model.
    """
    
    def explain_instance(
        self,
        instance: Union[pd.Series, pd.DataFrame],
        task_name: str,
        n_samples: int = 1000,
        n_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: Input instance to explain (Series or single-row DataFrame)
            task_name: Name of the task to explain
            n_samples: Number of samples to generate for LIME
            n_features: Number of top features to include in explanation
            
        Returns:
            Dict containing LIME explanation components
        """
        # Implementation details would include:
        # 1. Converting instance to the right format
        # 2. Creating a LIME explainer
        # 3. Generating a LIME explanation
        # 4. Processing the explanation into a standardized format
        
        # Placeholder for implementation
        self.logger.info("LIME explanation implementation will be added in a future update")
        
        # Convert instance to DataFrame if needed
        if isinstance(instance, pd.Series):
            instance_df = pd.DataFrame([instance])
        else:
            instance_df = instance
            
        # Get model prediction for this instance
        predictions = self.predictor.predict_dataframe(
            df=instance_df,
            task_names=[task_name],
            batch_size=1,
        )
        
        # Placeholder explanation
        feature_contributions = {
            feature: np.random.normal(0, 0.1) 
            for feature in self.feature_names[:n_features]
        }
        
        # Sort by absolute contribution
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        
        return {
            "task": task_name,
            "instance": instance,
            "prediction": predictions[task_name],
            "feature_contributions": dict(sorted_contributions),
            "intercept": 0.5,
            "method": "lime",
        }
    
    def plot_explanation(
        self,
        explanation: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot LIME explanation for a single instance.
        
        Args:
            explanation: Explanation dict from explain_instance
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Check that this is a LIME explanation
        if explanation.get("method") != "lime":
            raise ValueError("This is not a LIME explanation")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get feature contributions
        contributions = explanation["feature_contributions"]
        features = list(contributions.keys())
        values = list(contributions.values())
        
        # Create colors based on contribution direction
        colors = ["green" if v > 0 else "red" for v in values]
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel("Contribution")
        ax.set_title(f"LIME Explanation for {explanation['task']}")
        
        # Add prediction information
        pred_text = f"Prediction: {explanation['prediction']}"
        ax.annotate(
            pred_text,
            xy=(0.5, 0.02),
            xycoords="figure fraction",
            ha="center",
        )
        
        plt.tight_layout()
        return fig


class CounterfactualExplainer(LocalExplainer):
    """
    Counterfactual explainer for tabular transformer.
    
    This class generates counterfactual examples to explain what changes
    would be needed to alter a prediction.
    """
    
    def generate_counterfactual(
        self,
        instance: Union[pd.Series, pd.DataFrame],
        task_name: str,
        target_outcome: Any,
        features_to_vary: Optional[List[str]] = None,
        max_iterations: int = 1000,
        step_size: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Generate a counterfactual example.
        
        Args:
            instance: Input instance to explain (Series or single-row DataFrame)
            task_name: Name of the task to find counterfactual for
            target_outcome: Desired outcome to achieve
            features_to_vary: Optional list of features that can be varied (all by default)
            max_iterations: Maximum number of iterations for counterfactual search
            step_size: Step size for gradient-based search
            
        Returns:
            Dict containing counterfactual explanation
        """
        # Implementation details would include:
        # 1. Converting instance to the right format
        # 2. Setting up the optimization problem
        # 3. Running gradient-based search for counterfactual
        # 4. Processing the counterfactual into a standardized format
        
        # Placeholder for implementation
        self.logger.info("Counterfactual generation will be added in a future update")
        
        # Convert instance to DataFrame if needed
        if isinstance(instance, pd.Series):
            instance_df = pd.DataFrame([instance])
        else:
            instance_df = instance
            
        # Get original prediction
        original_pred = self.predictor.predict_dataframe(
            df=instance_df,
            task_names=[task_name],
            batch_size=1,
        )
        
        # Create a placeholder counterfactual by randomly perturbing features
        counterfactual = instance_df.copy()
        
        # Determine which features to vary
        if features_to_vary is None:
            features_to_vary = self.feature_names
        
        # Randomly perturb features
        for feature in features_to_vary:
            if feature in self.preprocessor.numeric_columns:
                # Perturb numeric feature
                counterfactual[feature] += np.random.normal(0, 0.5)
            elif feature in self.preprocessor.categorical_columns:
                # For categorical, we'd need to change to a different category
                # This is just a placeholder
                pass
        
        # Get counterfactual prediction
        counterfactual_pred = self.predictor.predict_dataframe(
            df=counterfactual,
            task_names=[task_name],
            batch_size=1,
        )
        
        # Calculate feature differences
        differences = {}
        for feature in features_to_vary:
            if feature in self.preprocessor.numeric_columns:
                orig_val = instance_df[feature].values[0]
                cf_val = counterfactual[feature].values[0]
                differences[feature] = {
                    "original": orig_val,
                    "counterfactual": cf_val,
                    "difference": cf_val - orig_val,
                }
        
        return {
            "task": task_name,
            "original_instance": instance,
            "counterfactual_instance": counterfactual.iloc[0],
            "original_prediction": original_pred[task_name],
            "counterfactual_prediction": counterfactual_pred[task_name],
            "target_outcome": target_outcome,
            "feature_differences": differences,
            "method": "counterfactual",
        }
    
    def plot_counterfactual(
        self,
        counterfactual_explanation: Dict[str, Any],
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot counterfactual explanation.
        
        Args:
            counterfactual_explanation: Explanation dict from generate_counterfactual
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Check that this is a counterfactual explanation
        if counterfactual_explanation.get("method") != "counterfactual":
            raise ValueError("This is not a counterfactual explanation")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get feature differences
        differences = counterfactual_explanation["feature_differences"]
        features = list(differences.keys())
        
        # Extract original and counterfactual values
        original_vals = [differences[f]["original"] for f in features]
        cf_vals = [differences[f]["counterfactual"] for f in features]
        
        # Set up plot
        x = np.arange(len(features))
        width = 0.35
        
        # Plot bars
        ax.bar(x - width/2, original_vals, width, label="Original")
        ax.bar(x + width/2, cf_vals, width, label="Counterfactual")
        
        # Add labels and legend
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_title("Counterfactual Explanation")
        ax.legend()
        
        # Add prediction information
        orig_pred = counterfactual_explanation["original_prediction"]
        cf_pred = counterfactual_explanation["counterfactual_prediction"]
        target = counterfactual_explanation["target_outcome"]
        
        pred_text = (
            f"Original Prediction: {orig_pred}\n"
            f"Counterfactual Prediction: {cf_pred}\n"
            f"Target Outcome: {target}"
        )
        
        ax.annotate(
            pred_text,
            xy=(0.5, 0.02),
            xycoords="figure fraction",
            ha="center",
        )
        
        plt.tight_layout()
        return fig
    
    def explain_instance(
        self,
        instance: Union[pd.Series, pd.DataFrame],
        task_name: str,
        target_outcome: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single instance using counterfactuals.
        
        Args:
            instance: Input instance to explain (Series or single-row DataFrame)
            task_name: Name of the task to explain
            target_outcome: Optional target outcome for counterfactual (will be inferred if not provided)
            
        Returns:
            Dict containing explanation components
        """
        # Convert instance to DataFrame if needed
        if isinstance(instance, pd.Series):
            instance_df = pd.DataFrame([instance])
        else:
            instance_df = instance
            
        # Get original prediction
        original_pred = self.predictor.predict_dataframe(
            df=instance_df,
            task_names=[task_name],
            batch_size=1,
        )
        
        # Infer target outcome if not provided
        if target_outcome is None:
            # This is a placeholder - for real implementation we'd pick a meaningful alternative
            target_outcome = "alternative_outcome"
        
        # Generate counterfactual
        return self.generate_counterfactual(
            instance=instance,
            task_name=task_name,
            target_outcome=target_outcome,
        )
    
    def plot_explanation(
        self,
        explanation: Dict[str, Any],
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot explanation for a single instance.
        
        Args:
            explanation: Explanation dict from explain_instance
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        return self.plot_counterfactual(explanation, figsize)
