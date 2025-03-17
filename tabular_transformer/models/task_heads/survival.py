"""
Survival analysis task head for tabular transformer.

This module implements a survival analysis head for the tabular transformer model,
capable of predicting survival probability over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from lifelines.utils import concordance_index

from tabular_transformer.models.task_heads.base import BaseTaskHead


class SurvivalHead(BaseTaskHead):
    """
    Survival analysis task head for tabular transformer.
    
    This head predicts survival curves for right-censored time-to-event data.
    It uses a discrete time approach by discretizing time into bins and
    predicting the conditional probability of survival at each time point.
    """
    
    def __init__(
        self,
        name: str,
        input_dim: int,
        num_time_bins: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        time_bin_boundaries: Optional[List[float]] = None,
    ):
        """
        Initialize survival analysis head.
        
        Args:
            name: Name of the task
            input_dim: Input dimension from encoder
            num_time_bins: Number of time bins for discrete-time survival model
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            time_bin_boundaries: Optional list of time boundaries for bins
        """
        super().__init__(
            name=name,
            input_dim=input_dim,
            output_dim=num_time_bins,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
        self.num_time_bins = num_time_bins
        self.time_bin_boundaries = time_bin_boundaries
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the survival head.
        
        Args:
            x: Input tensor from encoder [batch_size, input_dim]
            
        Returns:
            Logits tensor for hazard probabilities [batch_size, num_time_bins]
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
        Compute negative log-likelihood loss for survival prediction.
        
        Args:
            predictions: Logits from model [batch_size, num_time_bins]
            targets: Target tensor with two columns - event time and event indicator
                [batch_size, 2]
            mask: Optional mask for missing targets (1 = valid, 0 = missing)
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Returns:
            Loss tensor
        """
        # Extract time and event indicator from targets
        # targets[:, 0] is the time to event/censoring
        # targets[:, 1] is the event indicator (1 = event, 0 = censored)
        event_times = targets[:, 0]
        event_indicators = targets[:, 1]
        
        # Convert event times to discrete time bins
        time_bins = self._time_to_bin(event_times)
        
        # Compute negative log-likelihood for each sample
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Initialize loss tensor
        all_losses = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            time_bin = time_bins[i]
            event = event_indicators[i]
            
            # Get hazard probabilities (using sigmoid to convert logits to probabilities)
            hazards = torch.sigmoid(predictions[i, :])
            
            # If event=1, the probability of survival until time_bin and then failing
            # If event=0, the probability of survival until time_bin (censored)
            if time_bin >= self.num_time_bins:
                # If time is beyond the last bin, treat as censored at the last bin
                time_bin = self.num_time_bins - 1
                event = 0
            
            if event > 0:
                # For events, compute log probability of surviving until time_bin
                # and failing at time_bin
                if time_bin > 0:
                    # Log probability of surviving until time_bin
                    log_surv_prob = torch.sum(torch.log(1 - hazards[:time_bin]))
                else:
                    log_surv_prob = 0.0
                
                # Log probability of failing at time_bin
                log_hazard_prob = torch.log(hazards[time_bin] + 1e-7)
                
                # Negative log-likelihood
                all_losses[i] = -(log_surv_prob + log_hazard_prob)
            else:
                # For censored, compute log probability of surviving until time_bin
                if time_bin >= 0:
                    log_surv_prob = torch.sum(torch.log(1 - hazards[:time_bin + 1]))
                    all_losses[i] = -log_surv_prob
        
        # Apply mask if provided
        if mask is not None:
            all_losses = all_losses * mask
        
        # Apply reduction
        if reduction == "mean":
            # If mask provided, compute mean over valid samples
            if mask is not None:
                return all_losses.sum() / mask.sum().clamp(min=1.0)
            else:
                return all_losses.mean()
        elif reduction == "sum":
            return all_losses.sum()
        else:  # 'none'
            return all_losses
    
    def _time_to_bin(self, times: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous time values to discrete time bins.
        
        Args:
            times: Tensor of time values
            
        Returns:
            Tensor of time bin indices
        """
        if self.time_bin_boundaries is not None:
            # Use predefined bin boundaries
            boundaries = torch.tensor(
                self.time_bin_boundaries, device=times.device
            )
            bins = torch.bucketize(times, boundaries) - 1
            return torch.clamp(bins, min=0, max=self.num_time_bins - 1)
        else:
            # Default approach: divide the range into equal bins
            # This is a simplistic approach and may need refinement for real world data
            max_time = torch.max(times).item()
            bin_size = max_time / self.num_time_bins
            bins = (times / bin_size).long()
            return torch.clamp(bins, min=0, max=self.num_time_bins - 1)
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate survival probability predictions.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Dict containing survival probability curve and cumulative hazard
        """
        # Get hazard probabilities
        hazard_logits = self.forward(x)
        hazards = torch.sigmoid(hazard_logits)
        
        batch_size = hazards.shape[0]
        
        # Compute survival probabilities at each time point
        # S(t) = exp(-H(t)) = product(1 - h(j)) for j=1 to t
        survival_curves = torch.ones((batch_size, self.num_time_bins + 1), device=hazards.device)
        survival_curves[:, 1:] = torch.cumprod(1 - hazards, dim=1)
        
        # Compute cumulative hazard
        # H(t) = sum(-log(1 - h(j))) for j=1 to t
        cumulative_hazard = torch.zeros((batch_size, self.num_time_bins + 1), device=hazards.device)
        cumulative_hazard[:, 1:] = torch.cumsum(-torch.log(1 - hazards + 1e-7), dim=1)
        
        return {
            "hazards": hazards,
            "survival_curve": survival_curves,
            "cumulative_hazard": cumulative_hazard,
        }
    
    def predict_median_survival(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict median survival time.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Tensor of median survival times
        """
        predictions = self.predict(x)
        survival_curves = predictions["survival_curve"]
        
        # Find time bin where survival probability drops below 0.5
        median_bins = torch.argmax(survival_curves <= 0.5, dim=1)
        
        # Convert bin to time if time boundaries are provided
        if self.time_bin_boundaries is not None:
            # Add a final boundary for the last bin
            all_boundaries = self.time_bin_boundaries + [float('inf')]
            time_values = torch.tensor(
                all_boundaries, device=median_bins.device
            )[median_bins]
            return time_values
        else:
            # Without boundaries, just return the bin index as a proxy for time
            return median_bins.float()
            
    def evaluate(
        self,
        predictions: Union[Dict[str, torch.Tensor], pd.DataFrame],
        targets: Union[torch.Tensor, pd.Series, pd.DataFrame],
        metric: str = "c_index",
    ) -> float:
        """
        Evaluate survival model predictions against targets.
        
        Args:
            predictions: Dict of predictions from predict method or DataFrame from predict_dataframe
            targets: Target values (time and event status)
            metric: Evaluation metric ('c_index', 'integrated_brier_score')
            
        Returns:
            Performance score (for metrics like 'c_index', higher is better,
            but we negate the value to follow the convention that lower is better)
        """
        # Extract event times and event indicators from targets
        if isinstance(targets, pd.DataFrame):
            # Assume first column is time, second is event indicator
            if targets.shape[1] >= 2:
                time_values = targets.iloc[:, 0].values
                event_indicators = targets.iloc[:, 1].values
            else:
                # Try to find time and event columns by name
                if 'time' in targets.columns and 'event' in targets.columns:
                    time_values = targets['time'].values
                    event_indicators = targets['event'].values
                else:
                    raise ValueError("Could not identify time and event columns in targets DataFrame")
        elif isinstance(targets, pd.Series):
            # Single Series can't contain both time and event
            raise ValueError("Targets must contain both time and event information")
        elif isinstance(targets, torch.Tensor):
            # Assume tensor with 2 columns: [time, event]
            if targets.shape[1] == 2:
                time_values = targets[:, 0].cpu().numpy()
                event_indicators = targets[:, 1].cpu().numpy()
            else:
                raise ValueError("Target tensor must have 2 columns: [time, event]")
        else:
            # Try to convert to numpy array
            targets_array = np.array(targets)
            if targets_array.shape[1] == 2:
                time_values = targets_array[:, 0]
                event_indicators = targets_array[:, 1]
            else:
                raise ValueError("Targets must contain both time and event information")
        
        # Extract predicted risk scores from predictions
        if isinstance(predictions, pd.DataFrame):
            # For DataFrame output from predict_dataframe
            
            # Try to find risk scores or hazard values
            if any(col.startswith('cumulative_hazard_') for col in predictions.columns):
                # Use the last time point's cumulative hazard as risk score
                hazard_cols = [col for col in predictions.columns if col.startswith('cumulative_hazard_')]
                max_idx = max([int(col.split('_')[-1]) for col in hazard_cols])
                risk_scores = predictions[f'cumulative_hazard_{max_idx}'].values
            elif 'median_survival_0' in predictions.columns:
                # Lower median survival time means higher risk
                risk_scores = -predictions['median_survival_0'].values
            else:
                # Fallback to the first column
                risk_scores = predictions.iloc[:, 0].values
        else:
            # For dict output from predict method
            if "cumulative_hazard" in predictions:
                # Use last value of cumulative hazard as risk score
                risk_scores = predictions["cumulative_hazard"][:, -1].cpu().numpy()
            elif "hazards" in predictions:
                # Can use average hazard as risk score
                risk_scores = predictions["hazards"].mean(dim=1).cpu().numpy()
            else:
                # Try to use 1 - survival probability at final time point
                risk_scores = (1 - predictions["survival_curve"][:, -1]).cpu().numpy()
        
        # Ensure all values are numpy arrays
        time_values = np.array(time_values, dtype=np.float64)
        event_indicators = np.array(event_indicators, dtype=np.int64)
        risk_scores = np.array(risk_scores, dtype=np.float64)
        
        # Calculate the requested metric
        if metric == "c_index" or metric == "default":
            # Concordance index (C-index)
            # Higher values are better (range: 0-1)
            # Requires: event times, event indicators, risk scores
            score = concordance_index(time_values, risk_scores, event_indicators)
            # Negate score to follow convention that lower is better
            return -score
        
        elif metric == "integrated_brier_score":
            # Integrated Brier Score
            # Would need to implement this or use an external package
            # Lower values are better
            # For now, return a placeholder and warning
            print("Warning: Integrated Brier Score not fully implemented, returning placeholder")
            return 0.5
        
        elif metric == "log_likelihood":
            # Negative log-likelihood
            # Lower values are better
            # Compute this by predicting survival probabilities at event times
            # This would be a custom implementation
            print("Warning: Log-likelihood score not fully implemented, returning placeholder")
            return 1.0
            
        else:
            raise ValueError(f"Unknown metric for survival analysis: {metric}")
