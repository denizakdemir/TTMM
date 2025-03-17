"""
Classification task head for tabular transformer.

This module implements a classification head for the tabular transformer model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
            
    def evaluate(
        self,
        predictions: Union[Dict[str, torch.Tensor], pd.DataFrame],
        targets: Union[torch.Tensor, pd.Series, pd.DataFrame],
        metric: str = "accuracy",
    ) -> float:
        """
        Evaluate model predictions against targets.
        
        Args:
            predictions: Dict of predictions from predict method or DataFrame from predict_dataframe
            targets: Target values
            metric: Evaluation metric ('accuracy', 'precision', 'recall', 'f1', 'auc')
            
        Returns:
            Performance score (for metrics like 'accuracy', higher is better,
            but we negate the value to follow the convention that lower is better)
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
        
        # Get prediction values - handle both dict and DataFrame formats
        if isinstance(predictions, pd.DataFrame):
            # Handle DataFrame format from predict_dataframe
            if 'predicted_class_0' in predictions.columns:
                # Get the predicted class column 
                pred_column = 'predicted_class_0'
                pred_values = predictions[pred_column].values
            elif any(col.startswith('probability_') for col in predictions.columns):
                # Binary classification with probabilities, take the highest prob class
                prob_columns = [col for col in predictions.columns if col.startswith('probability_')]
                if len(prob_columns) == 1:  # Binary with single column (sigmoid output)
                    pred_values = (predictions[prob_columns[0]].values > 0.5).astype(int)
                else:  # Multiple probability columns
                    pred_values = predictions[prob_columns].values.argmax(axis=1)
            else:
                # Fallback to first column
                pred_values = predictions.iloc[:, 0].values
        else:
            # Handle dict format from predict method
            if "predicted_class" in predictions:
                pred_values = predictions["predicted_class"]
            elif "probabilities" in predictions:
                # Take argmax of probabilities
                pred_values = predictions["probabilities"].argmax(dim=1)
            else:
                # Fallback to logits
                pred_values = predictions["logits"].argmax(dim=1)
        
        # Ensure pred_values is a numpy array
        if isinstance(pred_values, torch.Tensor):
            pred_values = pred_values.detach().cpu().numpy()
        else:
            # Already a numpy array or similar
            pred_values = np.array(pred_values)
            
        # Make sure arrays are 1D
        targets_array = targets_array.flatten()
        pred_values = pred_values.flatten()
        
        # Calculate metric
        if metric == "accuracy" or metric == "default":
            # Accuracy - fraction of correctly predicted labels (higher is better)
            score = accuracy_score(targets_array, pred_values)
            # Negate to follow convention that lower is better
            return -score
        elif metric == "precision":
            # Precision - fraction of true positives among predicted positives
            score = precision_score(targets_array, pred_values, average='weighted', zero_division=0)
            return -score
        elif metric == "recall":
            # Recall - fraction of true positives identified
            score = recall_score(targets_array, pred_values, average='weighted', zero_division=0)
            return -score
        elif metric == "f1":
            # F1 score - harmonic mean of precision and recall
            score = f1_score(targets_array, pred_values, average='weighted', zero_division=0)
            return -score
        elif metric == "auc":
            # AUC-ROC - area under the receiver operating characteristic curve
            # Only for binary classification
            if isinstance(predictions, pd.DataFrame):
                prob_columns = [col for col in predictions.columns if col.startswith('probability_')]
                if len(prob_columns) == 1:
                    probs = predictions[prob_columns[0]].values
                else:
                    # Take probability of positive class
                    probs = predictions[prob_columns[1]].values if len(prob_columns) > 1 else None
            else:
                # Get probabilities from dict
                probs = predictions["probabilities"][:, 1].cpu().numpy() if predictions["probabilities"].shape[1] > 1 else predictions["probabilities"].cpu().numpy()
            
            if probs is not None:
                # Only compute AUC if we have proper probability outputs
                score = roc_auc_score(targets_array, probs)
                return -score
            else:
                # Fall back to accuracy if proper probabilities aren't available
                score = accuracy_score(targets_array, pred_values)
                return -score
        else:
            raise ValueError(f"Unknown metric for classification: {metric}")
