"""
Count outcome task head for tabular transformer.

This module implements a count outcome head for the tabular transformer model,
supporting Poisson, Negative Binomial, and Binomial distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from tabular_transformer.models.task_heads.base import BaseTaskHead


class CountHead(BaseTaskHead):
    """
    Count outcome task head for tabular transformer.
    
    This head can model count data using various distributions:
    - Poisson: For unbounded count data
    - Negative Binomial: For overdispersed count data
    - Binomial: For count data with a known maximum
    """
    
    def __init__(
        self,
        name: str,
        input_dim: int,
        distribution: str = "poisson",
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        max_count: Optional[int] = None,
    ):
        """
        Initialize count outcome head.
        
        Args:
            name: Name of the task
            input_dim: Input dimension from encoder
            distribution: Distribution type ('poisson', 'negative_binomial', 'binomial')
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            max_count: Maximum count for binomial distribution (n trials)
        """
        self.distribution = distribution.lower()
        
        # Determine output dimensions based on distribution
        if self.distribution == "poisson":
            # Poisson: output the log rate parameter (λ)
            output_dim = 1
        elif self.distribution == "negative_binomial":
            # Negative Binomial: output log(μ) and log(α)
            # Where μ is the mean and α is the dispersion parameter
            output_dim = 2
        elif self.distribution == "binomial":
            # Binomial: output logit(p) - probability of success
            output_dim = 1
            if max_count is None:
                raise ValueError("max_count must be provided for binomial distribution")
            self.max_count = max_count
        else:
            raise ValueError(
                f"Unsupported distribution: {distribution}. "
                f"Choose from: 'poisson', 'negative_binomial', 'binomial'"
            )
        
        super().__init__(
            name=name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the count head.
        
        Args:
            x: Input tensor from encoder [batch_size, input_dim]
            
        Returns:
            Distribution parameters [batch_size, output_dim]
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
        Compute negative log-likelihood loss for count prediction.
        
        Args:
            predictions: Distribution parameters from model
            targets: Target count values [batch_size, 1]
            mask: Optional mask for missing targets (1 = valid, 0 = missing)
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Returns:
            Loss tensor
        """
        if self.distribution == "poisson":
            return self._compute_poisson_loss(predictions, targets, mask, reduction)
        elif self.distribution == "negative_binomial":
            return self._compute_negative_binomial_loss(predictions, targets, mask, reduction)
        elif self.distribution == "binomial":
            return self._compute_binomial_loss(predictions, targets, mask, reduction)
    
    def _compute_poisson_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute Poisson negative log-likelihood loss.
        
        Args:
            predictions: Log rate parameter (λ) [batch_size, 1]
            targets: Target count values [batch_size, 1]
            mask: Optional mask for missing targets
            reduction: Loss reduction method
            
        Returns:
            Poisson NLL loss
        """
        log_lambda = predictions.view(-1)
        targets = targets.view(-1)
        
        # Poisson log-likelihood: y*log(λ) - λ - log(y!)
        # For numerical stability, we use the log-space formulation
        log_factorial = torch.lgamma(targets + 1)
        log_likelihood = targets * log_lambda - torch.exp(log_lambda) - log_factorial
        
        # Convert to loss (negative log-likelihood)
        loss = -log_likelihood
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.view(-1)
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
    
    def _compute_negative_binomial_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute Negative Binomial negative log-likelihood loss.
        
        Args:
            predictions: Parameters [batch_size, 2]
                First column: log mean (μ)
                Second column: log dispersion (α)
            targets: Target count values [batch_size, 1]
            mask: Optional mask for missing targets
            reduction: Loss reduction method
            
        Returns:
            Negative Binomial NLL loss
        """
        # Extract parameters
        log_mu = predictions[:, 0].view(-1)
        log_alpha = predictions[:, 1].view(-1)
        targets = targets.view(-1)
        
        # Calculate negative binomial log-likelihood
        # NB(y|μ,α) = Γ(y+α⁻¹)/Γ(y+1)Γ(α⁻¹) * (α⁻¹/(α⁻¹+μ))^(α⁻¹) * (μ/(μ+α⁻¹))^y
        
        mu = torch.exp(log_mu)
        alpha = torch.exp(log_alpha)
        alpha_inv = 1.0 / alpha
        
        # Log-likelihood computation in log space to avoid numerical issues
        log_likelihood = (
            torch.lgamma(targets + alpha_inv)
            - torch.lgamma(targets + 1)
            - torch.lgamma(alpha_inv)
            + alpha_inv * torch.log(alpha_inv / (alpha_inv + mu))
            + targets * torch.log(mu / (mu + alpha_inv))
        )
        
        # Convert to loss (negative log-likelihood)
        loss = -log_likelihood
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.view(-1)
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
    
    def _compute_binomial_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute Binomial negative log-likelihood loss.
        
        Args:
            predictions: Logit of success probability [batch_size, 1]
            targets: Target count values [batch_size, 1]
            mask: Optional mask for missing targets
            reduction: Loss reduction method
            
        Returns:
            Binomial NLL loss
        """
        logit_p = predictions.view(-1)
        targets = targets.view(-1)
        n = self.max_count
        
        # Ensure targets are within valid range [0, n]
        if (targets < 0).any() or (targets > n).any():
            raise ValueError(f"Target counts must be in range [0, {n}]")
        
        # Binomial log PMF: log[C(n,k) * p^k * (1-p)^(n-k)]
        #                  = log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
        # where C(n,k) is the binomial coefficient
        
        # Convert logits to probabilities
        log_p = F.logsigmoid(logit_p)
        log_1_minus_p = F.logsigmoid(-logit_p)
        
        # Calculate log of binomial coefficient
        log_coef = (
            torch.lgamma(torch.tensor(n + 1, device=targets.device))
            - torch.lgamma(targets + 1)
            - torch.lgamma(torch.tensor(n + 1, device=targets.device) - targets)
        )
        
        # Compute log-likelihood
        log_likelihood = (
            log_coef
            + targets * log_p
            + (n - targets) * log_1_minus_p
        )
        
        # Convert to loss (negative log-likelihood)
        loss = -log_likelihood
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.view(-1)
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
        Generate predictions for count data.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Dict containing distribution parameters and expected counts
        """
        outputs = self.forward(x)
        
        if self.distribution == "poisson":
            log_lambda = outputs.view(-1, 1)
            lambda_val = torch.exp(log_lambda)
            
            return {
                "rate": lambda_val,
                "expected_count": lambda_val,
                "variance": lambda_val,
            }
        
        elif self.distribution == "negative_binomial":
            log_mu = outputs[:, 0:1]
            log_alpha = outputs[:, 1:2]
            
            mu = torch.exp(log_mu)
            alpha = torch.exp(log_alpha)
            
            return {
                "mean": mu,
                "dispersion": alpha,
                "expected_count": mu,
                "variance": mu + alpha * mu**2,
            }
        
        elif self.distribution == "binomial":
            logit_p = outputs.view(-1, 1)
            p = torch.sigmoid(logit_p)
            n = self.max_count
            
            return {
                "probability": p,
                "num_trials": torch.full_like(p, n),
                "expected_count": n * p,
                "variance": n * p * (1 - p),
            }
        
    def predict_count(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict expected count.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Expected count tensor
        """
        predictions = self.predict(x)
        return predictions["expected_count"]
        
    def evaluate(
        self,
        predictions: Union[Dict[str, torch.Tensor], pd.DataFrame],
        targets: Union[torch.Tensor, pd.Series, pd.DataFrame],
        metric: str = "mse",
    ) -> float:
        """
        Evaluate count model predictions against targets.
        
        Args:
            predictions: Dict of predictions from predict method or DataFrame from predict_dataframe
            targets: Target count values
            metric: Evaluation metric ('mse', 'rmse', 'mae', 'poisson_deviance', etc.)
            
        Returns:
            Performance score (lower is better for all metrics)
        """
        # Convert targets to numpy array if it's a pandas Series
        if isinstance(targets, pd.Series):
            targets_array = targets.values
        elif isinstance(targets, torch.Tensor):
            # If it's a tensor, move it to CPU and convert to numpy
            targets_array = targets.detach().cpu().numpy()
        else:
            # Otherwise, convert directly
            targets_array = np.array(targets)
        
        # Get predicted counts - handle both dict and DataFrame formats
        if isinstance(predictions, pd.DataFrame):
            # Handle DataFrame format from predict_dataframe
            if 'expected_count_0' in predictions.columns:
                # Get the expected count column 
                pred_values = predictions['expected_count_0'].values
            elif any(col.startswith('rate_') for col in predictions.columns):
                # For Poisson, rate = expected count
                rate_col = next(col for col in predictions.columns if col.startswith('rate_'))
                pred_values = predictions[rate_col].values
            elif any(col.startswith('mean_') for col in predictions.columns):
                # For negative binomial, mean = expected count
                mean_col = next(col for col in predictions.columns if col.startswith('mean_'))
                pred_values = predictions[mean_col].values
            elif any(col.startswith('probability_') for col in predictions.columns):
                # For binomial, expected count = n*p
                prob_col = next(col for col in predictions.columns if col.startswith('probability_'))
                p_values = predictions[prob_col].values
                # We need n (max_count) for binomial
                if hasattr(self, 'max_count'):
                    n = self.max_count
                    pred_values = n * p_values
                else:
                    # If we don't have max_count, just use the probabilities
                    pred_values = p_values
            else:
                # Fallback to first column
                pred_values = predictions.iloc[:, 0].values
        else:
            # Handle dict format from predict method
            if "expected_count" in predictions:
                pred_values = predictions["expected_count"].cpu().numpy().flatten()
            elif "rate" in predictions:  # Poisson
                pred_values = predictions["rate"].cpu().numpy().flatten()
            elif "mean" in predictions:  # Negative binomial
                pred_values = predictions["mean"].cpu().numpy().flatten()
            elif "probability" in predictions:  # Binomial
                p_values = predictions["probability"].cpu().numpy().flatten()
                n = self.max_count if hasattr(self, 'max_count') else 1
                pred_values = n * p_values
            else:
                # Try to get the first available value
                first_key = list(predictions.keys())[0]
                pred_values = predictions[first_key].cpu().numpy().flatten()
        
        # Ensure targets and predictions are 1D arrays
        targets_array = targets_array.flatten()
        pred_values = np.array(pred_values).flatten()
        
        # Make sure arrays have the same length
        if len(targets_array) != len(pred_values):
            raise ValueError(f"Target length ({len(targets_array)}) does not match prediction length ({len(pred_values)})")
        
        # Calculate the requested metric
        if metric == "mse" or metric == "default":
            # Mean Squared Error
            return np.mean((targets_array - pred_values) ** 2)
            
        elif metric == "rmse":
            # Root Mean Squared Error
            return np.sqrt(np.mean((targets_array - pred_values) ** 2))
            
        elif metric == "mae":
            # Mean Absolute Error
            return np.mean(np.abs(targets_array - pred_values))
            
        elif metric == "mape":
            # Mean Absolute Percentage Error
            # Avoid division by zero by adding a small epsilon to targets
            # Only calculate for non-zero targets
            non_zero_mask = targets_array != 0
            if np.sum(non_zero_mask) == 0:
                return np.inf  # All targets are zero
            return np.mean(np.abs((targets_array[non_zero_mask] - pred_values[non_zero_mask]) / 
                                   (targets_array[non_zero_mask] + 1e-7))) * 100
            
        elif metric == "poisson_deviance":
            # Poisson Deviance
            # D = 2 * sum(y_true * log(y_true / y_pred) - (y_true - y_pred))
            # Handle cases where y_true or y_pred is zero
            non_zero_mask = (targets_array > 0) & (pred_values > 0)
            
            if np.sum(non_zero_mask) == 0:
                # No valid points for deviance calculation
                return 0.0
                
            # Calculate deviance term only for non-zero entries
            deviance_terms = np.zeros_like(targets_array, dtype=float)
            
            # For non-zero entries
            y_true = targets_array[non_zero_mask]
            y_pred = pred_values[non_zero_mask]
            deviance_terms[non_zero_mask] = y_true * np.log(y_true / y_pred) - (y_true - y_pred)
            
            # For y_true = 0, the term simplifies to y_pred
            zero_true_mask = targets_array == 0
            deviance_terms[zero_true_mask] = pred_values[zero_true_mask]
            
            # Sum and multiply by 2
            return 2 * np.sum(deviance_terms) / len(targets_array)
            
        elif metric == "r2":
            # R-squared (coefficient of determination)
            # 1 - (sum of squared residuals / total sum of squares)
            # Note: Higher values are better for R2, but we negate it to follow
            # the convention that lower is better
            if np.var(targets_array) == 0:
                # If all targets are the same, R2 is undefined
                return np.inf
                
            ss_res = np.sum((targets_array - pred_values) ** 2)
            ss_tot = np.sum((targets_array - np.mean(targets_array)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            # Negate to follow convention that lower is better
            return -r2
            
        else:
            raise ValueError(f"Unknown metric for count regression: {metric}")
