"""
Classification task head for tabular transformer.

This module implements a classification head for the tabular transformer model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from tabular_transformer.models.task_heads.base import BaseTaskHead


class ClassificationHead(BaseTaskHead):
    """
    Classification task head for tabular transformer.
    
    This head outputs class probabilities for multi-class classification tasks.
    """
    
    def __init__(
        self,
        name: str,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize classification head.
        
        Args:
            name: Name of the task
            input_dim: Input dimension from encoder
            num_classes: Number of classes (output dimension)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            class_weights: Optional tensor of class weights for imbalanced datasets
        """
        super().__init__(
            name=name,
            input_dim=input_dim,
            output_dim=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.num_classes = num_classes
        self.class_weights = class_weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.
        
        Args:
            x: Input tensor from encoder [batch_size, input_dim]
            
        Returns:
            Logits tensor [batch_size, num_classes]
        """
        logits = self.network(x)
        return logits
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            predictions: Logits from model [batch_size, num_classes]
            targets: Target class indices [batch_size]
            mask: Optional mask for missing targets (1 = valid, 0 = missing)
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Returns:
            Loss tensor
        """
        # Get per-sample losses
        if predictions.size(-1) == 1:  # Binary classification
            # Convert to binary classification format
            logits = predictions.view(-1)
            targets = targets.float()
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", weight=self.class_weights
            )
        else:  # Multi-class classification
            # Ensure targets are 1D as required by cross_entropy
            targets_1d = targets.squeeze().long()
            loss = F.cross_entropy(
                predictions, targets_1d, reduction="none", weight=self.class_weights
            )
        
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
            Dict containing probabilities and predicted classes
        """
        logits = self.forward(x)
        
        if self.num_classes == 1 or self.num_classes == 2:
            # Binary classification
            if self.num_classes == 1:
                probs = torch.sigmoid(logits)
                predicted_class = (probs > 0.5).long()
            else:  # num_classes == 2
                probs = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(probs, dim=1)
            
            return {
                "logits": logits,
                "probabilities": probs,
                "predicted_class": predicted_class,
            }
        else:
            # Multi-class classification
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1)
            
            return {
                "logits": logits,
                "probabilities": probs,
                "predicted_class": predicted_class,
            }
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate class probabilities.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Probability tensor
        """
        logits = self.forward(x)
        
        if self.num_classes == 1:
            # Binary classification with single output
            probs = torch.sigmoid(logits)
            return torch.cat([1 - probs, probs], dim=1)
        else:
            # Multi-class or binary with two outputs
            return F.softmax(logits, dim=1)
