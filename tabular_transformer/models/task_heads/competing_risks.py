"""
Competing risks task head for tabular transformer.

This module implements a competing risks analysis head for the tabular transformer model,
capable of predicting the probability of multiple competing events over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from lifelines.utils import concordance_index

from tabular_transformer.models.task_heads.base import BaseTaskHead


class CompetingRisksHead(BaseTaskHead):
    """
    Competing risks analysis task head for tabular transformer.
    
    This head predicts cumulative incidence functions for multiple competing events
    with right-censored time-to-event data.
    """
    
    def __init__(
        self,
        name: str,
        input_dim: int,
        num_time_bins: int,
        num_risks: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        time_bin_boundaries: Optional[List[float]] = None,
    ):
        """
        Initialize competing risks analysis head.
        
        Args:
            name: Name of the task
            input_dim: Input dimension from encoder
            num_time_bins: Number of time bins for discrete-time model
            num_risks: Number of competing event types (excluding censoring)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            time_bin_boundaries: Optional list of time boundaries for bins
        """
        # Output dimension is num_time_bins * num_risks for cause-specific hazards
        super().__init__(
            name=name,
            input_dim=input_dim,
            output_dim=num_time_bins * num_risks,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
        self.num_time_bins = num_time_bins
        self.num_risks = num_risks
        self.time_bin_boundaries = time_bin_boundaries
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the competing risks head.
        
        Args:
            x: Input tensor from encoder [batch_size, input_dim]
            
        Returns:
            Logits tensor for cause-specific hazards 
            [batch_size, num_risks, num_time_bins]
        """
        # Get raw outputs
        raw_outputs = self.network(x)
        
        # Reshape to [batch_size, num_risks, num_time_bins]
        batch_size = x.shape[0]
        reshaped = raw_outputs.view(batch_size, self.num_risks, self.num_time_bins)
        
        return reshaped
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for competing risks prediction.
        
        Args:
            predictions: Logits from model [batch_size, num_risks, num_time_bins]
            targets: Target tensor with two columns - event time and event type
                [batch_size, 2]
                event type: 0 = censored, 1...num_risks = event types
            mask: Optional mask for missing targets (1 = valid, 0 = missing)
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Returns:
            Loss tensor
        """
        # Extract time and event type from targets
        # targets[:, 0] is the time to event/censoring
        # targets[:, 1] is the event type (0 = censored, 1...num_risks = event types)
        event_times = targets[:, 0]
        event_types = targets[:, 1].long()
        
        # Convert event times to discrete time bins
        time_bins = self._time_to_bin(event_times)
        
        # Compute negative log-likelihood for each sample
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Initialize loss tensor
        all_losses = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            time_bin = time_bins[i]
            event_type = event_types[i]
            
            # Convert logits to hazards using sigmoid
            hazards = torch.sigmoid(predictions[i])  # [num_risks, num_time_bins]
            
            # Handle cases where time bin exceeds the number of bins
            if time_bin >= self.num_time_bins:
                # If time is beyond the last bin, treat as censored at the last bin
                time_bin = self.num_time_bins - 1
                event_type = 0
            
            # Calculate overall survival function (probability of no event of any type)
            # S(t) = exp(-sum_k(H_k(t))) = product_k,j(1 - h_k,j)
            # where h_k,j is the hazard for risk k at time j
            if time_bin > 0:
                # Probability of surviving (no event of any type) until time_bin
                log_surv_prob = torch.sum(
                    torch.log(1 - hazards[:, :time_bin].reshape(-1) + 1e-7)
                )
            else:
                log_surv_prob = 0.0
            
            if event_type > 0:
                # If an event occurred, add log hazard for the specific event type
                # at the event time
                risk_idx = event_type - 1  # Convert to 0-based index
                if risk_idx < self.num_risks:
                    log_hazard_prob = torch.log(hazards[risk_idx, time_bin] + 1e-7)
                    all_losses[i] = -(log_surv_prob + log_hazard_prob)
                else:
                    # If event type exceeds the number of risks, treat as censored
                    if time_bin < self.num_time_bins:
                        # Add log survival probability for time_bin
                        log_surv_prob_t = torch.sum(
                            torch.log(1 - hazards[:, time_bin] + 1e-7)
                        )
                        all_losses[i] = -(log_surv_prob + log_surv_prob_t)
            else:
                # For censored observations, compute log probability of no event
                # until the censoring time
                if time_bin < self.num_time_bins:
                    # Add log survival probability for time_bin
                    log_surv_prob_t = torch.sum(
                        torch.log(1 - hazards[:, time_bin] + 1e-7)
                    )
                    all_losses[i] = -(log_surv_prob + log_surv_prob_t)
                else:
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
            max_time = torch.max(times).item()
            bin_size = max_time / self.num_time_bins
            bins = (times / bin_size).long()
            return torch.clamp(bins, min=0, max=self.num_time_bins - 1)
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate cumulative incidence function predictions.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Dict containing cause-specific hazards, overall survival,
            and cumulative incidence functions
        """
        # Get cause-specific hazard probabilities
        hazard_logits = self.forward(x)  # [batch_size, num_risks, num_time_bins]
        hazards = torch.sigmoid(hazard_logits)
        
        batch_size = hazards.shape[0]
        
        # Compute overall survival function 
        # S(t) = exp(-sum_k(H_k(t))) = product_j(product_k(1 - h_k,j))
        # where h_k,j is the hazard for risk k at time j
        overall_survival = torch.ones(
            (batch_size, self.num_time_bins + 1), device=hazards.device
        )
        
        for t in range(self.num_time_bins):
            # Get hazards at time t for all risks
            hazards_t = hazards[:, :, t]  # [batch_size, num_risks]
            # Product of (1 - hazard) across all risks
            survival_t = torch.prod(1 - hazards_t, dim=1)
            # Multiply by previous survival
            overall_survival[:, t + 1] = overall_survival[:, t] * survival_t
        
        # Compute cumulative incidence functions for each risk
        # F_k(t) = sum_j=1^t [ S(j-1) * h_k,j ]
        cumulative_incidence = torch.zeros(
            (batch_size, self.num_risks, self.num_time_bins + 1), device=hazards.device
        )
        
        for k in range(self.num_risks):
            for t in range(self.num_time_bins):
                # Get hazard for risk k at time t
                hazard_kt = hazards[:, k, t]
                # CIF increment: probability of surviving until t and then failing from risk k
                cif_incr = overall_survival[:, t] * hazard_kt
                # Add to previous CIF
                cumulative_incidence[:, k, t + 1] = (
                    cumulative_incidence[:, k, t] + cif_incr
                )
        
        return {
            "hazards": hazards,
            "overall_survival": overall_survival,
            "cumulative_incidence": cumulative_incidence,
        }
    
    def predict_risk(self, x: torch.Tensor, time_horizon: int = None) -> torch.Tensor:
        """
        Predict risk (cumulative incidence) at a specific time horizon.
        
        Args:
            x: Input tensor from encoder
            time_horizon: Time horizon for prediction (bin index)
                Default is the final time bin
            
        Returns:
            Tensor of risk probabilities for each event type
                [batch_size, num_risks]
        """
        predictions = self.predict(x)
        cumulative_incidence = predictions["cumulative_incidence"]
        
        if time_horizon is None or time_horizon >= self.num_time_bins:
            time_horizon = self.num_time_bins
        
        return cumulative_incidence[:, :, time_horizon]
        
    def evaluate(
        self,
        predictions: Union[Dict[str, torch.Tensor], pd.DataFrame],
        targets: Union[torch.Tensor, pd.Series, pd.DataFrame],
        metric: str = "cause_specific_c_index",
    ) -> float:
        """
        Evaluate competing risks model predictions against targets.
        
        Args:
            predictions: Dict of predictions from predict method or DataFrame from predict_dataframe
            targets: Target values (time and event type)
            metric: Evaluation metric ('cause_specific_c_index', 'integrated_brier_score')
            
        Returns:
            Performance score (for metrics like 'c_index', higher is better,
            but we negate the value to follow the convention that lower is better)
        """
        # Extract event times and event types from targets
        if isinstance(targets, pd.DataFrame):
            # Assume first column is time, second is event type
            if targets.shape[1] >= 2:
                time_values = targets.iloc[:, 0].values
                event_types = targets.iloc[:, 1].values
            else:
                # Try to find time and event columns by name
                if 'time' in targets.columns and 'event' in targets.columns:
                    time_values = targets['time'].values
                    event_types = targets['event'].values
                else:
                    raise ValueError("Could not identify time and event columns in targets DataFrame")
        elif isinstance(targets, pd.Series):
            # Single Series can't contain both time and event type
            raise ValueError("Targets must contain both time and event type information")
        elif isinstance(targets, torch.Tensor):
            # Assume tensor with 2 columns: [time, event]
            if targets.shape[1] == 2:
                time_values = targets[:, 0].cpu().numpy()
                event_types = targets[:, 1].cpu().numpy()
            else:
                raise ValueError("Target tensor must have 2 columns: [time, event]")
        else:
            # Try to convert to numpy array
            targets_array = np.array(targets)
            if targets_array.shape[1] == 2:
                time_values = targets_array[:, 0]
                event_types = targets_array[:, 1]
            else:
                raise ValueError("Targets must contain both time and event type information")
        
        # Extract predicted risk scores from predictions
        if isinstance(predictions, pd.DataFrame):
            # For DataFrame output from predict_dataframe
            
            # Find the cumulative incidence columns for different risks
            all_cif_cols = {}
            for k in range(1, self.num_risks + 1):  # Risks are 1-indexed in event_types
                risk_cols = [col for col in predictions.columns 
                            if col.startswith(f'cumulative_incidence_{k-1}_')]
                if risk_cols:
                    # Use the last time point's incidence as risk score
                    max_idx = max([int(col.split('_')[-1]) for col in risk_cols])
                    all_cif_cols[k] = predictions[f'cumulative_incidence_{k-1}_{max_idx}'].values
            
            if not all_cif_cols:
                # Fallback to first few columns assuming they're risk scores
                for k in range(1, self.num_risks + 1):
                    if k <= predictions.shape[1]:
                        all_cif_cols[k] = predictions.iloc[:, k-1].values
        else:
            # For dict output from predict method
            all_cif_cols = {}
            if "cumulative_incidence" in predictions:
                # Extract cumulative incidence at final time point for each risk
                cif = predictions["cumulative_incidence"][:, :, -1].cpu().numpy()
                for k in range(1, self.num_risks + 1):  # Risks are 1-indexed
                    if k-1 < cif.shape[1]:  # Check if this risk exists
                        all_cif_cols[k] = cif[:, k-1]
        
        # Ensure time_values and event_types are numpy arrays
        time_values = np.array(time_values, dtype=np.float64)
        event_types = np.array(event_types, dtype=np.int64)
        
        # Calculate the requested metric
        if metric == "cause_specific_c_index" or metric == "default":
            # Calculate cause-specific concordance index for each risk type
            all_cindex_scores = []
            
            for k in range(1, self.num_risks + 1):  # Risks are 1-indexed
                if k in all_cif_cols:
                    # Get risk scores for this risk type
                    risk_scores = all_cif_cols[k]
                    
                    # Create cause-specific event indicator (1 for this risk, 0 otherwise)
                    cause_indicators = (event_types == k).astype(int)
                    
                    # Only calculate if we have events of this type
                    if np.sum(cause_indicators) > 0:
                        # Calculate concordance index for this risk
                        try:
                            c_index = concordance_index(
                                time_values, 
                                risk_scores, 
                                cause_indicators
                            )
                            all_cindex_scores.append(c_index)
                        except Exception as e:
                            print(f"Warning: Could not calculate C-index for risk {k}: {e}")
            
            # Take mean of all risk-specific C-indices
            if all_cindex_scores:
                avg_cindex = np.mean(all_cindex_scores)
                # Negate score to follow convention that lower is better
                return -avg_cindex
            else:
                # If no valid scores were calculated
                return 0.0
        
        elif metric == "integrated_brier_score":
            # Integrated Brier Score
            # Would need to implement a competing risks specific version
            # Lower values are better
            # For now, return a placeholder and warning
            print("Warning: Integrated Brier Score for competing risks not fully implemented, returning placeholder")
            return 0.5
        
        elif metric == "cumulative_incidence_error":
            # Error in cumulative incidence prediction
            # Custom metric that could be implemented for competing risks
            # Lower values are better
            print("Warning: Cumulative incidence error metric not fully implemented, returning placeholder")
            return 1.0
            
        else:
            raise ValueError(f"Unknown metric for competing risks analysis: {metric}")
