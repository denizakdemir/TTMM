"""
Loss functions for tabular transformer.

This module provides custom loss functions for different task types
and handles the KL divergence term for variational inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss for variational inference.
    
    This computes the KL divergence between a variational distribution 
    (parameterized by mean and log variance) and a standard Gaussian prior.
    """
    
    def __init__(self, beta: float = 1.0):
        """
        Initialize KL divergence loss.
        
        Args:
            beta: Weight for KL divergence term (often called β in β-VAE)
        """
        super().__init__()
        self.beta = beta
    
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss.
        
        Args:
            mu: Mean tensor [batch_size, latent_dim]
            logvar: Log variance tensor [batch_size, latent_dim]
            
        Returns:
            KL divergence loss
        """
        # KL(N(μ, σ) || N(0, 1)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Apply beta weight
        return self.beta * kl_loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    
    This aggregates losses from multiple task heads, optionally 
    with task-specific weights and a KL divergence term for 
    variational inference.
    """
    
    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        kl_weight: float = 1.0,
    ):
        """
        Initialize multi-task loss.
        
        Args:
            task_weights: Dict mapping task names to loss weights
            kl_weight: Weight for KL divergence term (β in β-VAE)
        """
        super().__init__()
        self.task_weights = task_weights or {}
        self.kl_divergence = KLDivergenceLoss(beta=kl_weight)
    
    def forward(
        self,
        task_losses: Dict[str, torch.Tensor],
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined multi-task loss.
        
        Args:
            task_losses: Dict mapping task names to individual losses
            mu: Optional mean tensor for variational inference
            logvar: Optional log variance tensor for variational inference
            
        Returns:
            Dict with total loss and individual components
        """
        # Aggregate task losses with weights
        weighted_losses = {}
        total_task_loss = 0.0
        
        for task_name, loss in task_losses.items():
            weight = self.task_weights.get(task_name, 1.0)
            weighted_loss = weight * loss
            weighted_losses[f"{task_name}_weighted"] = weighted_loss
            total_task_loss = total_task_loss + weighted_loss
        
        # Compute KL divergence if variational
        kl_loss = None
        if mu is not None and logvar is not None:
            kl_loss = self.kl_divergence(mu, logvar)
            total_loss = total_task_loss + kl_loss
        else:
            total_loss = total_task_loss
        
        # Return total loss and components
        result = {
            "total_loss": total_loss,
            "task_loss": total_task_loss,
            **task_losses,
            **weighted_losses,
        }
        
        if kl_loss is not None:
            result["kl_loss"] = kl_loss
            
        return result


# Utility functions for specific loss types

def masked_loss(
    loss_fn,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Apply a loss function with optional masking for missing values.
    
    Args:
        loss_fn: Loss function that returns per-element losses
        predictions: Model predictions
        targets: Target values
        mask: Optional mask tensor (1 = present, 0 = missing)
        reduction: Loss reduction method ('none', 'mean', 'sum')
        
    Returns:
        Loss tensor
    """
    # Compute per-element losses
    losses = loss_fn(predictions, targets)
    
    # Apply mask if provided
    if mask is not None:
        losses = losses * mask
    
    # Apply reduction
    if reduction == "none":
        return losses
    elif reduction == "sum":
        return losses.sum()
    else:  # "mean"
        if mask is not None:
            # Compute mean over non-masked elements
            return losses.sum() / mask.sum().clamp(min=1.0)
        else:
            return losses.mean()


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute masked MSE loss for regression tasks.
    
    Args:
        predictions: Model predictions
        targets: Target values
        mask: Optional mask tensor (1 = present, 0 = missing)
        reduction: Loss reduction method
        
    Returns:
        MSE loss
    """
    return masked_loss(
        lambda p, t: F.mse_loss(p, t, reduction="none"),
        predictions,
        targets,
        mask,
        reduction,
    )


def masked_binary_cross_entropy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute masked binary cross-entropy loss.
    
    Args:
        predictions: Model predictions (logits)
        targets: Binary target values
        mask: Optional mask tensor (1 = present, 0 = missing)
        reduction: Loss reduction method
        pos_weight: Optional weights for positive class
        
    Returns:
        BCE loss
    """
    loss_fn = lambda p, t: F.binary_cross_entropy_with_logits(
        p, t, reduction="none", pos_weight=pos_weight
    )
    return masked_loss(loss_fn, predictions, targets, mask, reduction)


def masked_cross_entropy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute masked cross-entropy loss for classification.
    
    Args:
        predictions: Model predictions (logits)
        targets: Target class indices
        mask: Optional mask tensor (1 = present, 0 = missing)
        reduction: Loss reduction method
        weight: Optional class weights
        
    Returns:
        Cross-entropy loss
    """
    loss_fn = lambda p, t: F.cross_entropy(
        p, t, reduction="none", weight=weight
    )
    return masked_loss(loss_fn, predictions, targets, mask, reduction)
