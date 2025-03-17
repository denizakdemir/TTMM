"""
Trainer for tabular transformer.

This module provides a trainer class for training tabular transformer models,
including support for multi-task learning and variational inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import os
from tqdm.auto import tqdm

from tabular_transformer.utils.logger import LoggerMixin
from tabular_transformer.utils.config import ModelConfig
from tabular_transformer.models.transformer_encoder import TabularTransformer
from tabular_transformer.models.task_heads.base import BaseTaskHead
from tabular_transformer.data.dataset import TabularDataset
from tabular_transformer.training.losses import MultiTaskLoss
from tabular_transformer.training.utils import EarlyStopping, MetricTracker, get_optimizer, get_lr_scheduler


class Trainer(LoggerMixin):
    """
    Trainer for tabular transformer models.
    
    This class handles training, validation, and evaluation of 
    tabular transformer models with support for multi-task learning.
    """
    
    def __init__(
        self,
        encoder: TabularTransformer,
        task_head: Union[BaseTaskHead, Dict[str, BaseTaskHead]],
        config: Optional[ModelConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            encoder: Transformer encoder
            task_head: Dict mapping task names to task heads or a single task head
            config: Optional model configuration
            optimizer: Optional optimizer (will be created if not provided)
            lr_scheduler: Optional learning rate scheduler
            device: Device to use for training (will use CUDA if available if not provided)
        """
        self.encoder = encoder
        # Convert single task head to dict if needed
        if isinstance(task_head, BaseTaskHead):
            self.task_heads = nn.ModuleDict({"main": task_head})
        else:
            self.task_heads = nn.ModuleDict(task_head)
        self.config = config
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Move models to device
        self.encoder.to(self.device)
        for head in self.task_heads.values():
            head.to(self.device)
        
        # Set up optimizer
        if optimizer is None and config is not None:
            parameters = list(self.encoder.parameters())
            for head in self.task_heads.values():
                parameters.extend(head.parameters())
            
            self.optimizer = get_optimizer(
                parameters,
                optimizer_name="adamw",
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer
            
        self.lr_scheduler = lr_scheduler
        
        # Set up loss function
        if config is not None:
            task_weights = {
                task_name: task_config.weight
                for task_name, task_config in config.tasks.items()
            }
            kl_weight = config.transformer.beta if config.transformer.variational else 0.0
        else:
            task_weights = None
            kl_weight = 1.0
            
        self.criterion = MultiTaskLoss(
            task_weights=task_weights,
            kl_weight=kl_weight,
        )
        
        # Set up metric tracking
        self.metric_tracker = MetricTracker()
        self.current_epoch = 0
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training set
            val_loader: Optional DataLoader for validation set
            
        Returns:
            Dict with metrics for the epoch
        """
        self.encoder.train()
        for head in self.task_heads.values():
            head.train()
        
        self.metric_tracker.start_epoch()
        epoch_loss = 0.0
        task_losses = {f"train_{task_name}_loss": 0.0 for task_name in self.task_heads}
        
        # Training loop
        for batch in tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}"):
            # Move inputs to device
            numeric_features = batch["numeric_features"].to(self.device)
            numeric_mask = batch["numeric_mask"].to(self.device)
            categorical_features = batch["categorical_features"].to(self.device)
            categorical_mask = batch["categorical_mask"].to(self.device)
            
            # Collect targets for each task
            targets = {}
            masks = {}
            for task_name in self.task_heads:
                targets[task_name] = batch[f"target_{task_name}"].to(self.device)
                masks[task_name] = batch.get(f"target_mask_{task_name}", None)
                if masks[task_name] is not None:
                    masks[task_name] = masks[task_name].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass through encoder
            encoder_output = self.encoder(
                numeric_features=numeric_features,
                numeric_mask=numeric_mask,
                categorical_features=categorical_features,
                categorical_mask=categorical_mask,
            )
            
            # Handle variational case
            mu = None
            logvar = None
            if isinstance(encoder_output, tuple):
                z, mu, logvar = encoder_output
            else:
                z = encoder_output
            
            # Forward pass through each task head and compute losses
            batch_task_losses = {}
            for task_name, head in self.task_heads.items():
                # Forward pass
                predictions = head(z)
                
                # Compute loss
                loss = head.compute_loss(
                    predictions=predictions,
                    targets=targets[task_name],
                    mask=masks[task_name],
                )
                
                batch_task_losses[task_name] = loss
            
            # Combine losses and compute KL divergence if variational
            loss_dict = self.criterion(
                task_losses=batch_task_losses,
                mu=mu,
                logvar=logvar,
            )
            
            total_loss = loss_dict["total_loss"]
            
            # Backward pass and optimizer step
            total_loss.backward()
            self.optimizer.step()
            
            # Update running losses
            epoch_loss += total_loss.item()
            for task_name in self.task_heads:
                task_losses[f"train_{task_name}_loss"] += batch_task_losses[task_name].item()
        
        # Calculate average losses
        num_batches = len(train_loader)
        epoch_loss /= num_batches
        for task_name in self.task_heads:
            task_losses[f"train_{task_name}_loss"] /= num_batches
        
        # Update metrics
        metrics = {"train_loss": epoch_loss, **task_losses}
        
        # Run validation if provided
        if val_loader is not None:
            val_metrics = self.evaluate(val_loader, prefix="val")
            metrics.update(val_metrics)
        
        # Update learning rate scheduler if using validation loss
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics.get("val_loss", epoch_loss))
            else:
                self.lr_scheduler.step()
        
        # Update metrics and epoch counter
        self.metric_tracker.update(metrics)
        self.metric_tracker.end_epoch()
        self.current_epoch += 1
        
        return metrics
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 10,
        early_stopping_patience: Optional[int] = None,
        model_dir: Optional[str] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training set
            val_loader: Optional DataLoader for validation set
            num_epochs: Number of epochs to train
            early_stopping_patience: Optional patience for early stopping
            model_dir: Optional directory to save checkpoints
            checkpoint_interval: Optional interval for saving checkpoints
            
        Returns:
            Dict with training history and best metrics
        """
        # Set up early stopping if requested
        early_stopping = None
        if early_stopping_patience is not None and val_loader is not None:
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                monitor="val_loss",
                mode="min",
                verbose=True,
            )
        
        # Create model directory if needed
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
        
        # Training loop
        self.logger.info(f"Starting training for {num_epochs} epochs")
        try:
            for epoch in range(num_epochs):
                metrics = self.train_epoch(train_loader, val_loader)
                
                # Check early stopping
                if early_stopping is not None:
                    early_stop = early_stopping(metrics["val_loss"], self)
                    if early_stop:
                        self.logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                        # Load best weights
                        early_stopping.load_best_state(self)
                        break
                
                # Save checkpoint if requested
                if model_dir is not None and checkpoint_interval is not None:
                    if (epoch + 1) % checkpoint_interval == 0:
                        self.save_checkpoint(os.path.join(model_dir, f"checkpoint_epoch_{epoch + 1}.pt"))
            
            # Save final model if requested
            if model_dir is not None:
                self.save_checkpoint(os.path.join(model_dir, "model_final.pt"))
                
            self.logger.info("Training completed")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted")
        
        # Prepare results
        training_history = {}
        for name in self.metric_tracker.metrics:
            training_history[name] = self.metric_tracker.get_history(name)
        
        best_metrics = {}
        for name in self.metric_tracker.best_metrics:
            best_metrics[name] = self.metric_tracker.get_best(name)
        
        return {
            "training_history": training_history,
            "best_metrics": best_metrics,
            "last_metrics": metrics,
        }
    
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            prefix: Prefix for metric names
            
        Returns:
            Dict with evaluation metrics
        """
        self.encoder.eval()
        for head in self.task_heads.values():
            head.eval()
        
        eval_loss = 0.0
        task_losses = {f"{prefix}_{task_name}_loss": 0.0 for task_name in self.task_heads}
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating ({prefix})"):
                # Move inputs to device
                numeric_features = batch["numeric_features"].to(self.device)
                numeric_mask = batch["numeric_mask"].to(self.device)
                categorical_features = batch["categorical_features"].to(self.device)
                categorical_mask = batch["categorical_mask"].to(self.device)
                
                # Collect targets for each task
                targets = {}
                masks = {}
                for task_name in self.task_heads:
                    targets[task_name] = batch[f"target_{task_name}"].to(self.device)
                    masks[task_name] = batch.get(f"target_mask_{task_name}", None)
                    if masks[task_name] is not None:
                        masks[task_name] = masks[task_name].to(self.device)
                
                # Forward pass through encoder
                encoder_output = self.encoder(
                    numeric_features=numeric_features,
                    numeric_mask=numeric_mask,
                    categorical_features=categorical_features,
                    categorical_mask=categorical_mask,
                )
                
                # Handle variational case
                mu = None
                logvar = None
                if isinstance(encoder_output, tuple):
                    z, mu, logvar = encoder_output
                else:
                    z = encoder_output
                
                # Forward pass through each task head and compute losses
                batch_task_losses = {}
                for task_name, head in self.task_heads.items():
                    # Forward pass
                    predictions = head(z)
                    
                    # Compute loss
                    loss = head.compute_loss(
                        predictions=predictions,
                        targets=targets[task_name],
                        mask=masks[task_name],
                    )
                    
                    batch_task_losses[task_name] = loss
                
                # Combine losses and compute KL divergence if variational
                loss_dict = self.criterion(
                    task_losses=batch_task_losses,
                    mu=mu,
                    logvar=logvar,
                )
                
                total_loss = loss_dict["total_loss"]
                
                # Update running losses
                eval_loss += total_loss.item()
                for task_name in self.task_heads:
                    task_losses[f"{prefix}_{task_name}_loss"] += batch_task_losses[task_name].item()
        
        # Calculate average losses
        num_batches = len(data_loader)
        eval_loss /= num_batches
        for task_name in self.task_heads:
            task_losses[f"{prefix}_{task_name}_loss"] /= num_batches
        
        # Return metrics
        metrics = {f"{prefix}_loss": eval_loss, **task_losses}
        return metrics
    
    def predict(
        self,
        data_loader: torch.utils.data.DataLoader,
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for a dataset.
        
        Args:
            data_loader: DataLoader for prediction
            task_names: Optional list of task names to predict (all by default)
            
        Returns:
            Dict mapping task names to predictions
        """
        self.encoder.eval()
        for head in self.task_heads.values():
            head.eval()
        
        # Determine which tasks to predict
        if task_names is None:
            task_names = list(self.task_heads.keys())
        
        # Initialize prediction lists
        all_predictions = {task_name: [] for task_name in task_names}
        all_latents = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                # Move inputs to device
                numeric_features = batch["numeric_features"].to(self.device)
                numeric_mask = batch["numeric_mask"].to(self.device)
                categorical_features = batch["categorical_features"].to(self.device)
                categorical_mask = batch["categorical_mask"].to(self.device)
                
                # Forward pass through encoder
                encoder_output = self.encoder(
                    numeric_features=numeric_features,
                    numeric_mask=numeric_mask,
                    categorical_features=categorical_features,
                    categorical_mask=categorical_mask,
                )
                
                # Handle variational case
                if isinstance(encoder_output, tuple):
                    z, _, _ = encoder_output
                else:
                    z = encoder_output
                
                # Store latent representations
                all_latents.append(z.cpu())
                
                # Forward pass through each task head
                for task_name in task_names:
                    head = self.task_heads[task_name]
                    predictions = head.predict(z)
                    
                    # Move predictions to CPU
                    for key, value in predictions.items():
                        if isinstance(value, torch.Tensor):
                            predictions[key] = value.cpu()
                    
                    all_predictions[task_name].append(predictions)
        
        # Concatenate predictions
        results = {"latent_representations": torch.cat(all_latents, dim=0)}
        
        for task_name in task_names:
            # Combine batch predictions
            task_preds = all_predictions[task_name]
            combined = {}
            
            # Get all keys from the first batch
            keys = task_preds[0].keys()
            
            # Concatenate tensors for each key
            for key in keys:
                tensors = [pred[key] for pred in task_preds]
                combined[key] = torch.cat(tensors, dim=0)
            
            results[task_name] = combined
        
        return results
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
        """
        checkpoint = {
            "encoder_state_dict": self.encoder.state_dict(),
            "task_heads_state_dict": {name: head.state_dict() for name, head in self.task_heads.items()},
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "current_epoch": self.current_epoch,
            "metrics": self.metric_tracker.metrics,
            "best_metrics": self.metric_tracker.best_metrics,
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to the checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load encoder
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        
        # Load task heads
        for name, state_dict in checkpoint["task_heads_state_dict"].items():
            if name in self.task_heads:
                self.task_heads[name].load_state_dict(state_dict)
            else:
                self.logger.warning(f"Task head '{name}' in checkpoint not found in current model")
        
        # Load optimizer if available
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler if available
        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
        # Load training state
        self.current_epoch = checkpoint.get("current_epoch", 0)
        if "metrics" in checkpoint:
            self.metric_tracker.metrics = checkpoint["metrics"]
        if "best_metrics" in checkpoint:
            self.metric_tracker.best_metrics = checkpoint["best_metrics"]
        
        self.logger.info(f"Loaded checkpoint from {filepath}")
