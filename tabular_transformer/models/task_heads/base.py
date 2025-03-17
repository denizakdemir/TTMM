"""
Base task head for tabular transformer.

This module defines an abstract base class for task-specific heads
that can be used with the tabular transformer model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any

from tabular_transformer.utils.logger import LoggerMixin
from tabular_transformer.utils.config import TaskConfig


class BaseTaskHead(nn.Module, ABC, LoggerMixin):
    """
    Abstract base class for task heads.
    
    This class defines the interface for task-specific heads
    that can be connected to the transformer encoder.
    """
    
    def __init__(
        self,
        name: str,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize base task head.
        
        Args:
            name: Name of the task
            input_dim: Input dimension from encoder
            output_dim: Output dimension for task
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [input_dim // 2]
        self.dropout = dropout
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer (without activation - will be task-specific)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        """
        Forward pass through the task head.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Task-specific output
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        predictions: Any,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute task-specific loss.
        
        Args:
            predictions: Model predictions
            targets: Target values
            mask: Optional mask for missing targets (1 = valid, 0 = missing)
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Returns:
            Loss tensor
        """
        pass
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> Any:
        """
        Generate predictions from the model.
        
        This method should convert the raw model outputs into
        the appropriate format for evaluation or interpretation.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Task-specific predictions
        """
        pass

    def transfer_gradients(self, grad_x: torch.Tensor) -> torch.Tensor:
        """
        Transfer gradients from task head to encoder.
        
        This method is used in multi-task learning to propagate gradients
        from multiple task heads back to the shared encoder.
        
        Args:
            grad_x: Gradient tensor from task output with respect to encoder output
            
        Returns:
            Gradient tensor to apply to encoder output
        """
        # Default implementation simply passes gradients through
        return grad_x
