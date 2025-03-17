"""
Regression task head for tabular transformer.

This module implements a regression head for the tabular transformer model.
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from tabular_transformer.models.task_heads.base import BaseTaskHead


class RegressionHead(BaseTaskHead):
    """
    Regression task head for tabular transformer.
    
    This head outputs continuous values for regression tasks.
    It can optionally output uncertainty estimates.
    """
    
    def __init__(
        self,
        name: str,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        uncertainty: bool = False,
    ):
        """
        Initialize regression head.
        
        Args:
            name: Name of the task
            input_dim: Input dimension from encoder
            output_dim: Number of regression outputs
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            uncertainty: Whether to output uncertainty estimates
        """
        # If uncertainty estimation is enabled, double the output dimension
        # to predict both mean and log variance
        actual_output_dim = output_dim * 2 if uncertainty else output_dim
        
        super().__init__(
            name=name,
            input_dim=input_dim,
            output_dim=actual_output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
        self.original_output_dim = output_dim
        self.uncertainty = uncertainty
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the regression head.
        
        Args:
            x: Input tensor from encoder [batch_size, input_dim]
            
        Returns:
            Predictions tensor [batch_size, output_dim] or
            [batch_size, output_dim*2] if uncertainty=True
        """
        return self.network(x)
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute regression loss.
        
        Args:
            predictions: Output from model 
                [batch_size, output_dim] or [batch_size, output_dim*2]
            targets: Target values [batch_size, output_dim]
            mask: Optional mask for missing targets (1 = valid, 0 = missing)
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Returns:
            Loss tensor
        """
        if self.uncertainty:
            return self._compute_gaussian_nll(predictions, targets, mask, reduction)
        else:
            return self._compute_mse(predictions, targets, mask, reduction)
    
    def _compute_mse(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute mean squared error loss.
        
        Args:
            predictions: Predictions from model [batch_size, output_dim]
            targets: Target values [batch_size, output_dim]
            mask: Optional mask for missing targets (1 = valid, 0 = missing)
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Returns:
            MSE loss tensor
        """
        # Compute squared error for each element
        loss = F.mse_loss(predictions, targets, reduction="none")
        
        # For multi-dimensional outputs, sum across output dimensions
        if loss.dim() > 1:
            loss = loss.sum(dim=1)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
        
        # Apply reduction
        if reduction == "mean":
            # If mask provided, compute mean over valid samples
            if mask is not None:
                return loss.sum() / mask.sum().clamp(min=1.0)
            else:
                return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
    
    def _compute_gaussian_nll(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for Gaussian distribution.
        
        Args:
            predictions: Model outputs containing means and log variances
                [batch_size, output_dim*2]
            targets: Target values [batch_size, output_dim]
            mask: Optional mask for missing targets (1 = valid, 0 = missing)
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Returns:
            NLL loss tensor
        """
        # Split predictions into means and log variances
        means, log_vars = torch.split(
            predictions, self.original_output_dim, dim=1
        )
        
        # Compute negative log-likelihood
        # NLL = 0.5 * (log(2π) + log(σ²) + (x - μ)²/σ²)
        precision = torch.exp(-log_vars)
        squared_error = (targets - means) ** 2
        
        # Log likelihood per dimension
        log_likelihood = -0.5 * (
            torch.log(2 * torch.tensor(3.14159, device=predictions.device)) +
            log_vars +
            squared_error * precision
        )
        
        # Sum across output dimensions (if multiple)
        if log_likelihood.dim() > 1:
            log_likelihood = log_likelihood.sum(dim=1)
        
        # Convert to negative log-likelihood (loss)
        loss = -log_likelihood
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
        
        # Apply reduction
        if reduction == "mean":
            # If mask provided, compute mean over valid samples
            if mask is not None:
                return loss.sum() / mask.sum().clamp(min=1.0)
            else:
                return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate predictions from the model.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Dict containing predictions and optional uncertainty estimates
        """
        outputs = self.forward(x)
        
        if self.uncertainty:
            # Split into mean and log variance
            means, log_vars = torch.split(
                outputs, self.original_output_dim, dim=1
            )
            variances = torch.exp(log_vars)
            std_devs = torch.sqrt(variances)
            
            return {
                "mean": means,
                "variance": variances,
                "std_dev": std_devs,
                "lower_bound": means - 1.96 * std_devs,  # 95% confidence interval
                "upper_bound": means + 1.96 * std_devs,  # 95% confidence interval
            }
        else:
            return {"prediction": outputs}
    
    def evaluate(
        self,
        predictions: Union[Dict[str, torch.Tensor], pd.DataFrame],
        targets: Union[torch.Tensor, pd.Series],
        metric: str = "mse",
    ) -> float:
        """
        Evaluate model predictions against targets.
        
        Args:
            predictions: Dict of predictions from predict method or DataFrame from predict_dataframe
            targets: Target values
            metric: Evaluation metric ('mse', 'rmse', or 'mae')
            
        Returns:
            Performance score (lower is better for all metrics)
        """
        # Convert targets to numpy array if it's a pandas Series
        if isinstance(targets, pd.Series):
            targets_array = targets.values
        else:
            # If it's already a tensor, move it to CPU and convert to numpy
            targets_array = targets.detach().cpu().numpy()
        
        # Get prediction values - handle both dict and DataFrame formats
        if isinstance(predictions, pd.DataFrame):
            # Handle DataFrame format from predict_dataframe
            if self.uncertainty and 'mean_0' in predictions.columns:
                # For uncertainty models, use the mean column
                pred_values = predictions['mean_0'].values
            else:
                # For standard models, use the first available column
                # (could be 'prediction_0' or similar)
                pred_columns = [col for col in predictions.columns if col.startswith('prediction_')]
                if pred_columns:
                    pred_values = predictions[pred_columns[0]].values
                else:
                    # Fallback to first column
                    pred_values = predictions.iloc[:, 0].values
        else:
            # Handle dict format from predict method
            if self.uncertainty:
                pred_values = predictions["mean"]
            else:
                pred_values = predictions["prediction"]
        
        # Ensure pred_values is a numpy array
        if isinstance(pred_values, torch.Tensor):
            pred_values = pred_values.detach().cpu().numpy().flatten()
        else:
            # Already a numpy array or similar
            pred_values = np.array(pred_values).flatten()
            
        # Ensure targets is a numpy array
        targets_array = np.array(targets_array).flatten()
        
        # Calculate metric
        if metric == "mse":
            # Mean Squared Error
            return ((pred_values - targets_array) ** 2).mean()
        elif metric == "rmse":
            # Root Mean Squared Error
            return np.sqrt(((pred_values - targets_array) ** 2).mean())
        elif metric == "mae":
            # Mean Absolute Error
            return np.abs(pred_values - targets_array).mean()
        else:
            raise ValueError(f"Unknown metric: {metric}")
