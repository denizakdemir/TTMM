"""
Visualization utilities for tabular transformer explainability.

This module provides classes and functions for visualizing model behavior
and explanations through various plots and graphical representations.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from tabular_transformer.utils.logger import LoggerMixin
from tabular_transformer.inference.predict import Predictor
from tabular_transformer.data.dataset import TabularDataset


class ExplainabilityViz(LoggerMixin):
    """
    Base class for explainability visualizations.
    
    This class provides common functionality for creating
    visualizations to explain model behavior.
    """
    
    def __init__(
        self,
        predictor: Predictor,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize visualization utilities.
        
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
    
    def set_style(self, style: str = "whitegrid") -> None:
        """
        Set the visualization style.
        
        Args:
            style: Seaborn style name
        """
        sns.set_style(style)
    
    def create_multi_plot_figure(
        self,
        n_plots: int,
        n_cols: int = 2,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Tuple[Figure, List[plt.Axes]]:
        """
        Create a figure with multiple subplots.
        
        Args:
            n_plots: Number of plots
            n_cols: Number of columns
            figsize: Figure size
            
        Returns:
            Tuple of (figure, list of axes)
        """
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=figsize,
            squeeze=False,
        )
        
        # Hide unused subplots
        if n_plots < n_rows * n_cols:
            for i in range(n_plots, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
        
        return fig, axes.flatten()[:n_plots]


class PDPlot(ExplainabilityViz):
    """
    Partial Dependence Plot (PDP) for tabular transformer.
    
    This class generates plots showing the marginal effect of features
    on model predictions.
    """
    
    def compute_partial_dependence(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        feature: str,
        task_name: str,
        grid_points: int = 20,
        percentile_range: Tuple[float, float] = (0.05, 0.95),
        batch_size: int = 64,
    ) -> Dict[str, np.ndarray]:
        """
        Compute partial dependence for a feature.
        
        Args:
            data: Input data (DataFrame or TabularDataset)
            feature: Feature name to compute PDP for
            task_name: Name of the task to compute PDP for
            grid_points: Number of points in the grid
            percentile_range: Range of percentiles to cover
            batch_size: Batch size for prediction
            
        Returns:
            Dict containing grid values and partial dependence values
        """
        # Convert to DataFrame if needed
        if isinstance(data, TabularDataset):
            data_df = data.dataframe
        else:
            data_df = data
        
        # Check if feature is valid
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in feature names")
        
        # Check if task is valid
        if task_name not in self.task_heads:
            raise ValueError(f"Task '{task_name}' not found in model")
        
        # Determine feature type
        is_categorical = feature in self.preprocessor.categorical_columns
        
        if is_categorical:
            # For categorical features, use unique values
            unique_values = data_df[feature].dropna().unique()
            grid = np.sort(unique_values)
        else:
            # For numeric features, create a grid
            low = np.percentile(data_df[feature].dropna(), percentile_range[0] * 100)
            high = np.percentile(data_df[feature].dropna(), percentile_range[1] * 100)
            grid = np.linspace(low, high, grid_points)
        
        # Initialize array for partial dependence values
        pd_values = np.zeros(len(grid))
        
        # Iterate over grid points
        for i, value in enumerate(grid):
            # Create copies of the data with the feature set to the grid value
            modified_dfs = []
            
            # Process in batches to avoid memory issues
            for batch_start in range(0, len(data_df), batch_size):
                batch_end = min(batch_start + batch_size, len(data_df))
                batch_df = data_df.iloc[batch_start:batch_end].copy()
                batch_df[feature] = value
                modified_dfs.append(batch_df)
            
            # Combine batches
            modified_df = pd.concat(modified_dfs, axis=0)
            
            # Get predictions
            preds = self.predictor.predict_dataframe(
                df=modified_df,
                task_names=[task_name],
                batch_size=batch_size,
            )
            
            # Extract main prediction value (depends on task type)
            task_pred = preds[task_name]
            
            # Different task types have different prediction structures
            if "prediction" in task_pred.columns:
                # Typical structure for regression
                pd_values[i] = task_pred["prediction"].mean()
            elif "class_probs_1" in task_pred.columns:
                # Typical structure for binary classification (positive class probability)
                pd_values[i] = task_pred["class_probs_1"].mean()
            else:
                # Default to the first prediction column
                first_col = task_pred.columns[0]
                pd_values[i] = task_pred[first_col].mean()
        
        return {
            "feature": feature,
            "task": task_name,
            "grid": grid,
            "pd_values": pd_values,
            "values": pd_values,  # Add expected key
            "average_prediction": pd_values,  # Add expected key
            "is_categorical": is_categorical,
        }
    
    def plot_partial_dependence(
        self,
        pd_result: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot partial dependence.
        
        Args:
            pd_result: Result from compute_partial_dependence
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        feature = pd_result["feature"]
        task = pd_result["task"]
        grid = pd_result["grid"]
        pd_values = pd_result["pd_values"]
        is_categorical = pd_result["is_categorical"]
        
        # Create plot
        if is_categorical:
            # Bar plot for categorical features
            ax.bar(range(len(grid)), pd_values)
            ax.set_xticks(range(len(grid)))
            ax.set_xticklabels(grid, rotation=45, ha="right")
        else:
            # Line plot for numeric features
            ax.plot(grid, pd_values)
            ax.set_xlabel(feature)
        
        ax.set_ylabel(f"Partial Dependence ({task})")
        ax.set_title(f"Partial Dependence Plot for {feature}")
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_pdp(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        features: List[str],
        task_name: str,
        grid_points: int = 20,
        percentile_range: Tuple[float, float] = (0.05, 0.95),
        batch_size: int = 64,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Plot partial dependence for multiple features.
        
        Args:
            data: Input data (DataFrame or TabularDataset)
            features: List of feature names to compute PDP for
            task_name: Name of the task to compute PDP for
            grid_points: Number of points in the grid
            percentile_range: Range of percentiles to cover
            batch_size: Batch size for prediction
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Create multi-plot figure
        fig, axes = self.create_multi_plot_figure(
            n_plots=len(features),
            n_cols=2,
            figsize=figsize,
        )
        
        # Compute and plot PDP for each feature
        for i, feature in enumerate(features):
            pd_result = self.compute_partial_dependence(
                data=data,
                feature=feature,
                task_name=task_name,
                grid_points=grid_points,
                percentile_range=percentile_range,
                batch_size=batch_size,
            )
            
            # Extract data
            grid = pd_result["grid"]
            pd_values = pd_result["pd_values"]
            is_categorical = pd_result["is_categorical"]
            
            # Create plot
            ax = axes[i]
            if is_categorical:
                # Bar plot for categorical features
                ax.bar(range(len(grid)), pd_values)
                ax.set_xticks(range(len(grid)))
                ax.set_xticklabels(grid, rotation=45, ha="right")
            else:
                # Line plot for numeric features
                ax.plot(grid, pd_values)
                ax.set_xlabel(feature)
            
            ax.set_ylabel("Partial Dependence")
            ax.set_title(f"PDP for {feature}")
        
        plt.tight_layout()
        return fig


class ICEPlot(PDPlot):
    """
    Individual Conditional Expectation (ICE) Plot for tabular transformer.
    
    This class generates plots showing how predictions for individual samples
    change as a feature is varied.
    """
    
    def compute_ice_curves(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        feature: str,
        task_name: str,
        n_samples: int = 10,
        grid_points: int = 20,
        percentile_range: Tuple[float, float] = (0.05, 0.95),
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute Individual Conditional Expectation (ICE) curves.
        
        Args:
            data: Input data (DataFrame or TabularDataset)
            feature: Feature name to compute ICE for
            task_name: Name of the task to compute ICE for
            n_samples: Number of samples to include
            grid_points: Number of points in the grid
            percentile_range: Range of percentiles to cover
            random_state: Random seed for sample selection
            
        Returns:
            Dict containing grid values and ICE curves for each sample
        """
        # Convert to DataFrame if needed
        if isinstance(data, TabularDataset):
            data_df = data.dataframe
        else:
            data_df = data
        
        # Check if feature is valid
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in feature names")
        
        # Check if task is valid
        if task_name not in self.task_heads:
            raise ValueError(f"Task '{task_name}' not found in model")
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
        
        # Select random samples
        if n_samples < len(data_df):
            sample_indices = np.random.choice(len(data_df), n_samples, replace=False)
            samples = data_df.iloc[sample_indices].copy()
        else:
            samples = data_df.copy()
        
        # Determine feature type
        is_categorical = feature in self.preprocessor.categorical_columns
        
        if is_categorical:
            # For categorical features, use unique values
            unique_values = data_df[feature].dropna().unique()
            grid = np.sort(unique_values)
        else:
            # For numeric features, create a grid
            low = np.percentile(data_df[feature].dropna(), percentile_range[0] * 100)
            high = np.percentile(data_df[feature].dropna(), percentile_range[1] * 100)
            grid = np.linspace(low, high, grid_points)
        
        # Initialize array for ICE curves
        ice_values = np.zeros((len(samples), len(grid)))
        
        # Store original feature values for each sample
        original_values = samples[feature].values
        
        # Iterate over grid points
        for i, value in enumerate(grid):
            # Create copies of the samples with the feature set to the grid value
            modified_samples = samples.copy()
            modified_samples[feature] = value
            
            # Get predictions
            preds = self.predictor.predict_dataframe(
                df=modified_samples,
                task_names=[task_name],
                batch_size=len(modified_samples),
            )
            
            # Extract main prediction value (depends on task type)
            task_pred = preds[task_name]
            
            # Different task types have different prediction structures
            if "prediction" in task_pred.columns:
                # Typical structure for regression
                ice_values[:, i] = task_pred["prediction"].values
            elif "class_probs_1" in task_pred.columns:
                # Typical structure for binary classification (positive class probability)
                ice_values[:, i] = task_pred["class_probs_1"].values
            else:
                # Default to the first prediction column
                first_col = task_pred.columns[0]
                ice_values[:, i] = task_pred[first_col].values
        
        # Calculate partial dependence as the mean of ICE curves
        pd_values = np.mean(ice_values, axis=0)
        
        return {
            "feature": feature,
            "task": task_name,
            "grid": grid,
            "ice_values": ice_values,
            "ice_curves": ice_values,  # Add expected key
            "pd_values": pd_values,
            "pd_curve": pd_values,  # Add expected key
            "values": grid,  # Add expected key
            "sample_indices": sample_indices if n_samples < len(data_df) else np.arange(len(data_df)),
            "original_values": original_values,
            "instances": sample_indices if n_samples < len(data_df) else np.arange(len(data_df)),  # Add expected key
            "is_categorical": is_categorical,
        }
    
    def plot_ice_curves(
        self,
        ice_result: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 6),
        alpha: float = 0.3,
        center: bool = True,
    ) -> plt.Figure:
        """
        Plot Individual Conditional Expectation (ICE) curves.
        
        Args:
            ice_result: Result from compute_ice_curves
            figsize: Figure size
            alpha: Transparency for individual curves
            center: Whether to center curves at the original value
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        feature = ice_result["feature"]
        task = ice_result["task"]
        grid = ice_result["grid"]
        ice_values = ice_result["ice_values"]
        pd_values = ice_result["pd_values"]
        original_values = ice_result["original_values"]
        is_categorical = ice_result["is_categorical"]
        
        # Center curves if requested
        if center and not is_categorical:
            # Find the grid point closest to each original value
            original_indices = []
            for val in original_values:
                idx = np.abs(grid - val).argmin()
                original_indices.append(idx)
            
            # Center curves
            centered_ice = np.zeros_like(ice_values)
            for i in range(len(ice_values)):
                centered_ice[i] = ice_values[i] - ice_values[i, original_indices[i]]
            
            # Centered PD
            centered_pd = np.mean(centered_ice, axis=0)
            
            # Plot centered ICE curves
            for i in range(len(centered_ice)):
                ax.plot(grid, centered_ice[i], alpha=alpha, color="blue")
            
            # Plot centered PD
            ax.plot(grid, centered_pd, color="red", linewidth=2, label="Mean Effect")
            
            ax.set_ylabel(f"Centered Effect on {task}")
        else:
            # Plot ICE curves
            for i in range(len(ice_values)):
                ax.plot(grid, ice_values[i], alpha=alpha, color="blue")
            
            # Plot PD
            ax.plot(grid, pd_values, color="red", linewidth=2, label="Mean Effect")
            
            ax.set_ylabel(f"Effect on {task}")
        
        # Set labels and title
        if is_categorical:
            ax.set_xticks(range(len(grid)))
            ax.set_xticklabels(grid, rotation=45, ha="right")
        else:
            ax.set_xlabel(feature)
        
        ax.set_title(f"ICE Plot for {feature}")
        ax.legend()
        
        plt.tight_layout()
        return fig


class CalibrationPlot(ExplainabilityViz):
    """
    Calibration plots for tabular transformer.
    
    This class generates plots comparing predicted probabilities
    against observed outcomes.
    """
    
    def compute_calibration(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        task_name: str,
        target_column: str,
        n_bins: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Compute calibration curve.
        
        Args:
            data: Input data (DataFrame or TabularDataset)
            task_name: Name of the task to compute calibration for
            target_column: Column name for the target variable
            n_bins: Number of bins for calibration
            batch_size: Batch size for prediction
            
        Returns:
            Dict containing calibration data
        """
        # Convert to DataFrame if needed
        if isinstance(data, TabularDataset):
            data_df = data.dataframe
        else:
            data_df = data
        
        # Check if task is valid
        if task_name not in self.task_heads:
            raise ValueError(f"Task '{task_name}' not found in model")
        
        # Get predictions
        preds = self.predictor.predict_dataframe(
            df=data_df,
            task_names=[task_name],
            batch_size=batch_size,
        )
        
        # Extract target
        y_true = data_df[target_column].values
        
        # Extract prediction probabilities (depends on task type)
        task_pred = preds[task_name]
        
        # Different task types have different prediction structures
        if "class_probs_1" in task_pred.columns:
            # Binary classification
            y_pred = task_pred["class_probs_1"].values
        elif "prediction" in task_pred.columns:
            # Regression/probability output
            y_pred = task_pred["prediction"].values
        else:
            # Default to the first prediction column
            first_col = task_pred.columns[0]
            y_pred = task_pred[first_col].values
        
        # Create bins and compute calibration curve
        bins = np.linspace(0, 1, n_bins + 1)
        binned_y_pred = np.digitize(y_pred, bins) - 1
        
        # Ensure we don't have negative indices (for predictions below the lowest bin boundary)
        binned_y_pred = np.clip(binned_y_pred, 0, n_bins - 1)
        
        bin_counts = np.bincount(binned_y_pred, minlength=n_bins)
        
        # Avoid division by zero
        bin_counts = np.maximum(bin_counts, 1)
        
        # Compute mean predicted value and fraction of positives in each bin
        bin_sums = np.zeros(n_bins)
        bin_true = np.zeros(n_bins)
        
        for i in range(len(y_pred)):
            bin_idx = binned_y_pred[i]
            bin_sums[bin_idx] += y_pred[i]
            bin_true[bin_idx] += y_true[i]
        
        mean_predicted = bin_sums / bin_counts
        fraction_positive = bin_true / bin_counts
        
        return {
            "task": task_name,
            "mean_predicted": mean_predicted,
            "prob_pred": mean_predicted,  # Add expected key
            "fraction_positive": fraction_positive,
            "prob_true": fraction_positive,  # Add expected key
            "bin_counts": bin_counts,
            "hist": bin_counts,  # Add expected key
            "n_bins": n_bins,
        }
    
    def plot_calibration_curve(
        self,
        calibration_result: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Plot calibration curve.
        
        Args:
            calibration_result: Result from compute_calibration_curve
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Create figure with 2 subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        
        # Main calibration plot
        ax1 = fig.add_subplot(gs[0])
        
        # Extract data
        task = calibration_result["task"]
        mean_predicted = calibration_result["mean_predicted"]
        fraction_positive = calibration_result["fraction_positive"]
        bin_counts = calibration_result["bin_counts"]
        n_bins = calibration_result["n_bins"]
        
        # Plot calibration curve
        ax1.plot(
            mean_predicted,
            fraction_positive,
            "s-",
            label=f"{task} Calibration Curve",
        )
        
        # Plot perfect calibration line
        ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly Calibrated")
        
        # Set labels and title
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_title("Calibration Curve")
        ax1.legend(loc="lower right")
        
        # Plot histogram of predictions
        ax2 = fig.add_subplot(gs[1])
        ax2.bar(
            np.linspace(0, 1, n_bins, endpoint=False) + 0.5/n_bins,
            bin_counts,
            width=1/n_bins,
            alpha=0.8,
        )
        ax2.set_xlabel("Mean Predicted Probability")
        ax2.set_ylabel("Count")
        ax2.set_title("Histogram of Predictions")
        
        plt.tight_layout()
        return fig
