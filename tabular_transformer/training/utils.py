"""
Training utilities for tabular transformer.

This module provides utility functions for training, including
early stopping, metrics, and other helper functions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import datetime
from collections import defaultdict

from tabular_transformer.utils.logger import LoggerMixin


class EarlyStopping(LoggerMixin):
    """
    Early stopping callback to prevent overfitting.
    
    This monitors a validation metric and stops training
    when the metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        monitor: str = "val_loss",
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement after which training stops
            min_delta: Minimum change in monitored value to qualify as improvement
            monitor: Metric to monitor
            mode: 'min' or 'max' - whether to minimize or maximize the monitored value
            verbose: Whether to log early stopping events
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None
        
        if mode == "min":
            self.val_score_improved = lambda score, best_score: (
                score < best_score - min_delta
            )
        elif mode == "max":
            self.val_score_improved = lambda score, best_score: (
                score > best_score + min_delta
            )
        else:
            raise ValueError(f"Mode {mode} is not supported. Use 'min' or 'max'.")
    
    def __call__(
        self, 
        score: float, 
        model: torch.nn.Module,
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current score to evaluate
            model: Model whose state should be saved if score improves
            
        Returns:
            True if early stopping criterion is met
        """
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(model)
        elif self.val_score_improved(score, self.best_score):
            # Score improved
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            # Score did not improve
            self.counter += 1
            if self.verbose:
                self.logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(self, model) -> None:
        """
        Save model state when validation score improves.
        
        Args:
            model: Model to save (can be a Trainer instance or nn.Module)
        """
        if self.verbose:
            self.logger.info(
                f"Validation score improved ({self.best_score:.6f}). Saving model state."
            )
            
        # Check if model is a Trainer instance
        from tabular_transformer.training.trainer import Trainer
        if isinstance(model, Trainer):
            # Save encoder and task heads states
            self.best_state_dict = {
                'encoder': {k: v.cpu().clone() for k, v in model.encoder.state_dict().items()},
                'task_heads': {name: {k: v.cpu().clone() for k, v in head.state_dict().items()} 
                              for name, head in model.task_heads.items()}
            }
        else:
            # Standard PyTorch model
            self.best_state_dict = {
                key: value.cpu().clone() for key, value in model.state_dict().items()
            }
    
    def load_best_state(self, model) -> None:
        """
        Load the best model state.
        
        Args:
            model: Model to update with best state (can be a Trainer instance or nn.Module)
        """
        if self.best_state_dict is not None:
            # Check if model is a Trainer instance
            from tabular_transformer.training.trainer import Trainer
            if isinstance(model, Trainer) and isinstance(self.best_state_dict, dict) and 'encoder' in self.best_state_dict:
                # Load encoder state
                model.encoder.load_state_dict(self.best_state_dict['encoder'])
                
                # Load task heads states
                for name, head in model.task_heads.items():
                    if name in self.best_state_dict['task_heads']:
                        head.load_state_dict(self.best_state_dict['task_heads'][name])
            else:
                # Standard PyTorch model
                model.load_state_dict(self.best_state_dict)
                
            if self.verbose:
                self.logger.info("Loaded best model state.")


class MetricTracker(LoggerMixin):
    """
    Track and log training metrics.
    
    This keeps track of metrics during training and validation,
    and provides functionality to log and retrieve them.
    """
    
    def __init__(self):
        """Initialize metric tracker."""
        self.metrics = defaultdict(list)
        self.epoch_metrics = {}
        self.best_metrics = {}
        self.start_time = None
    
    def start_epoch(self) -> None:
        """Mark the start of a new epoch."""
        self.epoch_metrics = {}
        self.start_time = time.time()
    
    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update metrics for the current epoch.
        
        Args:
            metrics: Dict mapping metric names to values
        """
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.epoch_metrics[name] = value
    
    def end_epoch(self) -> Dict[str, float]:
        """
        Mark the end of an epoch and log metrics.
        
        Returns:
            Dict of metrics for the epoch
        """
        elapsed = time.time() - self.start_time
        self.epoch_metrics["time"] = elapsed
        
        # Update history
        for name, value in self.epoch_metrics.items():
            self.metrics[name].append(value)
            
            # Update best metrics
            if name not in self.best_metrics:
                self.best_metrics[name] = value
            elif "loss" in name:
                # For losses, lower is better
                self.best_metrics[name] = min(self.best_metrics[name], value)
            elif "accuracy" in name or "score" in name or "auc" in name:
                # For accuracy/scores, higher is better
                self.best_metrics[name] = max(self.best_metrics[name], value)
        
        # Log epoch metrics
        metric_str = " | ".join(
            f"{k}: {v:.4f}" for k, v in self.epoch_metrics.items()
            if k != "time"
        )
        time_str = str(datetime.timedelta(seconds=int(elapsed)))
        self.logger.info(f"Metrics: {metric_str} | time: {time_str}")
        
        return self.epoch_metrics
    
    def get_latest(self, name: str) -> float:
        """
        Get the most recent value for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Latest value of the metric
        """
        if name in self.epoch_metrics:
            return self.epoch_metrics[name]
        elif name in self.metrics and len(self.metrics[name]) > 0:
            return self.metrics[name][-1]
        else:
            raise KeyError(f"Metric {name} not found")
    
    def get_history(self, name: str) -> List[float]:
        """
        Get the history of a metric.
        
        Args:
            name: Metric name
            
        Returns:
            List of values for the metric across epochs
        """
        if name in self.metrics:
            return self.metrics[name]
        else:
            raise KeyError(f"Metric {name} not found")
    
    def get_best(self, name: str) -> float:
        """
        Get the best value for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Best value of the metric
        """
        if name in self.best_metrics:
            return self.best_metrics[name]
        else:
            raise KeyError(f"Metric {name} not found")


def get_optimizer(
    model_params: List[torch.Tensor],
    optimizer_name: str = "adam",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create an optimizer instance.
    
    Args:
        model_params: Model parameters to optimize
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd', etc.)
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer parameters
        
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model_params, lr=learning_rate, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            model_params, lr=learning_rate, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model_params, lr=learning_rate, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            model_params, lr=learning_rate, weight_decay=weight_decay, **kwargs
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "reduce_on_plateau",
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler
        **kwargs: Additional scheduler parameters
        
    Returns:
        Scheduler instance or None if not requested
    """
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None
    
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
    elif scheduler_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
