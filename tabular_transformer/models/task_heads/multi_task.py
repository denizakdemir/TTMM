
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any

from tabular_transformer.models.task_heads.base import BaseTaskHead


class MultiTaskHead(BaseTaskHead):
    """
    Multi-task head for tabular transformer.
    
    This head manages multiple task heads and handles the distribution of
    data and aggregation of losses across all tasks.
    """
    
    def __init__(
        self,
        name: str = "multi_task",
        input_dim: int = None,
        output_dim: int = 0,  # Not directly used as output dimension varies by task
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        heads: Dict[str, BaseTaskHead] = None,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-task head.
        
        Args:
            name: Name of the multi-task head
            input_dim: Input dimension from encoder (must match all heads)
            output_dim: Not directly used, but required by BaseTaskHead
            hidden_dims: Not directly used as each head has its own hidden layers
            dropout: Dropout probability for potential shared layers
            heads: Dictionary mapping task names to task heads
            task_weights: Optional dictionary mapping task names to loss weights
        """
        # Determine input_dim from the first head if not provided
        if input_dim is None and heads is not None and len(heads) > 0:
            first_head = next(iter(heads.values()))
            if hasattr(first_head, 'input_dim'):
                input_dim = first_head.input_dim
            
        # Use a default if still None
        if input_dim is None:
            input_dim = 128  # Default value
            
        super().__init__(name=name, 
                         input_dim=input_dim, 
                         output_dim=output_dim,
                         hidden_dims=hidden_dims,
                         dropout=dropout)
        
        # Store the dictionary of task heads and their names
        self._task_heads = nn.ModuleDict() if heads is None else nn.ModuleDict(heads)
        self._task_head_names = list(self._task_heads.keys()) if heads is not None else []
        
        # Validate all heads have the same input dimension
        for head_name, head in self._task_heads.items():
            if head.input_dim != input_dim:
                raise ValueError(
                    f"Head '{head_name}' has input_dim {head.input_dim}, "
                    f"which doesn't match expected {input_dim}"
                )
        
        # Set default weights if not provided
        if task_weights is None:
            self._task_weights = {name: 1.0 for name in self._task_head_names}
        else:
            self._task_weights = task_weights.copy() if task_weights else {}
            # Fill in missing weights with default value
            for name in self._task_head_names:
                if name not in self._task_weights:
                    self._task_weights[name] = 1.0
    
    # Add property getters to avoid name collisions
    @property
    def heads(self) -> nn.ModuleDict:
        return self._task_heads
        
    @property
    def task_head_names(self) -> List[str]:
        return self._task_head_names
        
    @property
    def task_weights(self) -> Dict[str, float]:
        return self._task_weights
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all task heads.
        
        Args:
            x: Input tensor from encoder [batch_size, input_dim]
            
        Returns:
            Dictionary mapping task names to task outputs
        """
        outputs = {}
        for name, head in self._task_heads.items():
            outputs[name] = head(x)
        return outputs
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss across all tasks.
        
        Args:
            predictions: Dictionary mapping task names to prediction tensors
            targets: Dictionary mapping task names to target tensors
            masks: Optional dictionary mapping task names to mask tensors
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Returns:
            Tuple of (total loss, individual task losses)
        """
        task_losses = {}
        
        # Compute loss for each task
        for name in self._task_head_names:
            head = self._task_heads[name]
            pred = predictions[name]
            target = targets[name]
            mask = None if masks is None else masks.get(name, None)
            
            # Compute task-specific loss
            task_loss = head.compute_loss(pred, target, mask, reduction)
            task_losses[name] = task_loss
        
        # Compute weighted sum of task losses
        if not task_losses:
            return torch.tensor(0.0, requires_grad=True), {}
            
        total_loss = torch.zeros(1, device=next(iter(task_losses.values())).device)
        for name, loss in task_losses.items():
            weight = self._task_weights.get(name, 1.0)
            total_loss = total_loss + weight * loss
        
        return total_loss, task_losses
    
    def predict(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate predictions from all task heads.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Nested dictionary mapping task names to prediction dictionaries
        """
        predictions = {}
        for name, head in self._task_heads.items():
            predictions[name] = head.predict(x)
        return predictions
