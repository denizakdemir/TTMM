"""
Global explainability methods for tabular transformer.

This module provides classes and functions for explaining model behavior
at a global level, focusing on feature importance and overall model behavior.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from tabular_transformer.utils.logger import LoggerMixin
from tabular_transformer.models.transformer_encoder import TabularTransformer
from tabular_transformer.inference.predict import Predictor
from tabular_transformer.data.dataset import TabularDataset


class GlobalExplainer(LoggerMixin):
    """
    Base class for global explainability methods.
    
    This class provides common functionality for explaining model behavior
    at a global level, focusing on feature importance and overall patterns.
    """
    
    def __init__(
        self,
        predictor: Predictor,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize global explainer.
        
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
    
    def plot_feature_importance(
        self,
        importance_scores: Dict[str, np.ndarray],
        task_name: Optional[str] = None,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Plot feature importance scores.
        
        Args:
            importance_scores: Dict mapping task names to feature importance arrays
            task_name: Specific task to plot (plots all tasks if None)
            top_n: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if task_name is not None:
            # Plot for a specific task
            if task_name not in importance_scores:
                raise ValueError(f"Task '{task_name}' not found in importance scores")
            
            scores = importance_scores[task_name]
            task_names = [task_name]
        else:
            # Plot for all tasks
            task_names = list(importance_scores.keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            nrows=len(task_names),
            ncols=1,
            figsize=figsize,
            squeeze=False,
        )
        
        for i, task in enumerate(task_names):
            # Get scores for this task
            scores = importance_scores[task]
            
            # Create DataFrame with feature names and scores
            df = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": scores,
            })
            
            # Sort by importance and take top N
            df = df.sort_values("Importance", ascending=False).head(top_n)
            
            # Plot horizontal bar chart
            ax = axes[i, 0]
            sns.barplot(x="Importance", y="Feature", data=df, ax=ax)
            ax.set_title(f"Feature Importance for {task}")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
        
        plt.tight_layout()
        return fig


class PermutationImportance(GlobalExplainer):
    """
    Permutation feature importance for tabular transformer.
    
    This class measures the importance of features by permuting their values
    and measuring the change in model performance.
    """
    
    def compute_importance(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        target_columns: Dict[str, str],
        task_names: Optional[List[str]] = None,
        n_repeats: int = 5,
        batch_size: int = 64,
        random_state: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute permutation feature importance.
        
        Args:
            data: Input data (DataFrame or TabularDataset)
            target_columns: Dict mapping task names to target column names in data
            task_names: List of task names to compute importance for (all by default)
            n_repeats: Number of times to repeat permutation
            batch_size: Batch size for prediction
            random_state: Random seed for permutation
            
        Returns:
            Dict mapping task names to arrays of feature importance scores
        """
        # Determine which tasks to use
        if task_names is None:
            task_names = list(self.task_heads.keys())
            
        # Ensure all specified tasks exist in the model
        for task_name in task_names:
            if task_name not in self.task_heads:
                raise ValueError(f"Task '{task_name}' not found in model")
            
        # Ensure we have targets for all tasks
        for task_name in task_names:
            if task_name not in target_columns:
                raise ValueError(f"No target column specified for task '{task_name}'")
        
        # Convert to DataFrame if needed
        if isinstance(data, TabularDataset):
            data_df = data.dataframe
        else:
            data_df = data
            
        # Get baseline predictions and performance
        self.logger.info("Computing baseline predictions...")
        baseline_preds = self.predictor.predict_dataframe(
            df=data_df,
            task_names=task_names,
            batch_size=batch_size,
        )
        
        # Calculate baseline performance
        baseline_perf = {}
        for task_name in task_names:
            head = self.task_heads[task_name]
            preds = baseline_preds[task_name]
            target = data_df[target_columns[task_name]]
            baseline_perf[task_name] = head.evaluate(preds, target)
                
        # Initialize importance scores
        importance_scores = {
            task_name: np.zeros(len(self.feature_names))
            for task_name in task_names
        }
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
        
        # Iterate over features
        for feat_idx, feature in enumerate(tqdm(self.feature_names, desc="Computing Permutation Importance")):
            # Track performance drops for this feature
            perf_drops = {task_name: [] for task_name in task_names}
            
            # Repeat permutation n_repeats times
            for i in range(n_repeats):
                # Create a copy of the data
                permuted_df = data_df.copy()
                
                # Permute the feature
                permuted_df[feature] = np.random.permutation(permuted_df[feature].values)
                
                # Get predictions with permuted feature
                permuted_preds = self.predictor.predict_dataframe(
                    df=permuted_df,
                    task_names=task_names,
                    batch_size=batch_size,
                )
                
                # Calculate performance drops
                for task_name in task_names:
                    head = self.task_heads[task_name]
                    preds = permuted_preds[task_name]
                    target = data_df[target_columns[task_name]]
                    permuted_perf = head.evaluate(preds, target)
                    
                    # Calculate performance drop (higher is more important)
                    perf_drop = baseline_perf[task_name] - permuted_perf
                    perf_drops[task_name].append(perf_drop)
            
            # Average performance drops over repeats
            for task_name in task_names:
                importance_scores[task_name][feat_idx] = np.mean(perf_drops[task_name])
        
        return importance_scores


class IntegratedGradients(GlobalExplainer):
    """
    Integrated gradients for tabular transformer.
    
    This class computes feature importance by integrating gradients
    along a straight line from a baseline to the input.
    """
    
    def compute_importance(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        task_names: Optional[List[str]] = None,
        n_steps: int = 50,
        batch_size: int = 64,
        baseline: Optional[Union[str, pd.DataFrame]] = "zero",
    ) -> Dict[str, np.ndarray]:
        """
        Compute integrated gradients feature importance.
        
        Args:
            data: Input data (DataFrame or TabularDataset)
            task_names: List of task names to compute importance for (all by default)
            n_steps: Number of steps in the integration
            batch_size: Batch size for prediction
            baseline: Baseline for integration ("zero", "mean", or custom DataFrame)
            
        Returns:
            Dict mapping task names to arrays of feature importance scores
        """
        # Implementation details would include:
        # 1. Creating baseline inputs
        # 2. Interpolating between baseline and original inputs
        # 3. Computing gradients at each step
        # 4. Integrating gradients to get feature importance
        
        # Placeholder for implementation
        self.logger.info("Integrated Gradients implementation will be added in a future update")
        
        # Placeholder return
        if task_names is None:
            task_names = list(self.task_heads.keys())
            
        return {
            task_name: np.ones(len(self.feature_names))
            for task_name in task_names
        }


class SHAPExplainer(GlobalExplainer):
    """
    SHAP (SHapley Additive exPlanations) for tabular transformer.
    
    This class computes SHAP values to explain feature importance.
    """
    
    def compute_importance(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        task_names: Optional[List[str]] = None,
        n_samples: int = 100,
        batch_size: int = 64,
    ) -> Dict[str, np.ndarray]:
        """
        Compute SHAP feature importance.
        
        Args:
            data: Input data (DataFrame or TabularDataset)
            task_names: List of task names to compute importance for (all by default)
            n_samples: Number of samples for SHAP approximation
            batch_size: Batch size for prediction
            
        Returns:
            Dict mapping task names to arrays of feature importance scores
        """
        # Implementation details would include:
        # 1. Setting up a SHAP explainer (e.g., KernelExplainer or DeepExplainer)
        # 2. Computing SHAP values for each feature
        # 3. Aggregating values across samples
        
        # Placeholder for implementation
        self.logger.info("SHAP implementation will be added in a future update")
        
        # Placeholder return
        if task_names is None:
            task_names = list(self.task_heads.keys())
            
        return {
            task_name: np.ones(len(self.feature_names))
            for task_name in task_names
        }


class AttentionExplainer(GlobalExplainer):
    """
    Attention-based explainability for tabular transformer.
    
    This class extracts and analyzes attention weights from the transformer
    to explain feature importance.
    """
    
    def extract_attention_weights(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        batch_size: int = 64,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from transformer for given data.
        
        Args:
            data: Input data (DataFrame or TabularDataset)
            batch_size: Batch size for processing
            
        Returns:
            Dict mapping attention layer names to attention weight tensors
        """
        # Implementation details would include:
        # 1. Modifying the transformer encoder to capture attention weights
        # 2. Running a forward pass with hooks to collect weights
        # 3. Formatting and returning the attention matrices
        
        # Placeholder for implementation
        self.logger.info("Attention weight extraction will be added in a future update")
        
        # Placeholder return
        return {}
    
    def compute_importance(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        batch_size: int = 64,
        aggregation: str = "mean",
    ) -> Dict[str, np.ndarray]:
        """
        Compute feature importance based on attention weights.
        
        Args:
            data: Input data (DataFrame or TabularDataset)
            batch_size: Batch size for processing
            aggregation: Method to aggregate attention across heads/layers ("mean", "max", "sum")
            
        Returns:
            Dict mapping "attention" to array of feature importance scores
        """
        # Implementation details would include:
        # 1. Extracting attention weights
        # 2. Aggregating weights across heads and layers
        # 3. Normalizing to get feature importance scores
        
        # Placeholder for implementation
        self.logger.info("Attention-based importance will be added in a future update")
        
        # Placeholder return
        return {
            "attention": np.ones(len(self.feature_names))
        }
