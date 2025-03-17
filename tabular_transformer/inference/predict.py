"""
Prediction utilities for tabular transformer.

This module provides classes and functions for generating predictions
from trained tabular transformer models.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm

from tabular_transformer.utils.logger import LoggerMixin
from tabular_transformer.models.transformer_encoder import TabularTransformer
from tabular_transformer.models.task_heads.base import BaseTaskHead
from tabular_transformer.data.preprocess import FeaturePreprocessor
from tabular_transformer.data.dataset import TabularDataset


class Predictor(LoggerMixin):
    """
    Predictor for tabular transformer models.
    
    This class handles inference on new data using a trained
    tabular transformer model with one or more task heads.
    """
    
    def __init__(
        self,
        encoder: TabularTransformer,
        task_head: Dict[str, BaseTaskHead],  # Changed from task_heads to task_head
        preprocessor: FeaturePreprocessor,
        device: Optional[str] = None,
    ):
        """
        Initialize predictor.
        
        Args:
            encoder: Trained transformer encoder
            task_head: Dict mapping task names to trained task heads
            preprocessor: Fitted feature preprocessor
            device: Device to use for prediction (will use CUDA if available if not provided)
        """
        self.encoder = encoder
        self.task_heads = task_head  # Store as task_heads internally
        self.preprocessor = preprocessor
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Move models to device
        self.encoder.to(self.device)
        for head in self.task_heads.values():
            head.to(self.device)
        
        # Set evaluation mode
        self.encoder.eval()
        for head in self.task_heads.values():
            head.eval()
    
    def predict(
        self,
        data: Union[pd.DataFrame, TabularDataset, torch.utils.data.DataLoader],
        task_names: Optional[List[str]] = None,
        batch_size: int = 64,
        include_latent: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate predictions for new data.
        
        Args:
            data: Input data (DataFrame, TabularDataset, or DataLoader)
            task_names: Optional list of task names to predict (all by default)
            batch_size: Batch size for prediction (ignored if DataLoader is provided)
            include_latent: Whether to include latent representations in results
            
        Returns:
            Dict mapping task names to prediction results
        """
        # Determine which tasks to predict
        if task_names is None:
            task_names = list(self.task_heads.keys())
        else:
            # Validate task names
            for task_name in task_names:
                if task_name not in self.task_heads:
                    raise ValueError(f"Task '{task_name}' not found in model")
        
        # Create DataLoader if needed
        if isinstance(data, pd.DataFrame):
            # Create TabularDataset from DataFrame
            dataset = TabularDataset(
                dataframe=data,
                numeric_columns=self.preprocessor.numeric_columns,
                categorical_columns=self.preprocessor.categorical_columns,
                preprocessor=self.preprocessor,
                fit_preprocessor=False,
            )
            data_loader = dataset.create_dataloader(
                batch_size=batch_size, shuffle=False, num_workers=0
            )
        elif isinstance(data, TabularDataset):
            # Create DataLoader from TabularDataset
            data_loader = data.create_dataloader(
                batch_size=batch_size, shuffle=False, num_workers=0
            )
        elif isinstance(data, torch.utils.data.DataLoader):
            # Use provided DataLoader
            data_loader = data
        else:
            raise TypeError(
                "data must be a pandas DataFrame, TabularDataset, or DataLoader"
            )
        
        # Initialize prediction storage
        all_predictions = {task_name: [] for task_name in task_names}
        all_latents = [] if include_latent else None
        
        # Generate predictions
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
                
                # Store latent representations if requested
                if include_latent:
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
        
        # Combine predictions from batches
        results = {}
        
        # Include latent representations if requested
        if include_latent:
            results["latent_representations"] = torch.cat(all_latents, dim=0).numpy()
        
        # Combine task-specific predictions
        for task_name in task_names:
            # Get all prediction batches for this task
            task_preds = all_predictions[task_name]
            
            if not task_preds:
                continue
                
            # Get all keys from the first batch
            keys = task_preds[0].keys()
            
            # Combine predictions for each key
            combined = {}
            for key in keys:
                tensors = [pred[key] for pred in task_preds]
                combined[key] = torch.cat(tensors, dim=0).numpy()
            
            results[task_name] = combined
        
        return results
    
    def predict_dataframe(
        self,
        df: pd.DataFrame,
        task_names: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate predictions and return as DataFrames.
        
        Args:
            df: Input DataFrame
            task_names: Optional list of task names to predict (all by default)
            batch_size: Batch size for prediction
            
        Returns:
            Dict mapping task names to prediction DataFrames
        """
        # Get predictions as NumPy arrays
        predictions = self.predict(
            data=df,
            task_names=task_names,
            batch_size=batch_size,
            include_latent=False,
        )
        
        # Convert to DataFrames
        dataframes = {}
        for task_name, task_preds in predictions.items():
            # Create a DataFrame for this task's predictions
            df_pred = pd.DataFrame(index=df.index)
            
            # Add each prediction component as columns
            for key, value in task_preds.items():
                # Handle differently based on shape
                if value.ndim == 1:
                    # Single column
                    df_pred[f"{key}"] = value
                else:
                    # Multiple columns
                    for i in range(value.shape[1]):
                        df_pred[f"{key}_{i}"] = value[:, i]
            
            dataframes[task_name] = df_pred
        
        return dataframes
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        encoder: TabularTransformer,
        task_head: Dict[str, BaseTaskHead],  # Changed from task_heads to task_head
        preprocessor: FeaturePreprocessor,
        device: Optional[str] = None,
    ) -> "Predictor":
        """
        Create predictor from a checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            encoder: Transformer encoder with matching architecture
            task_head: Dict mapping task names to task heads with matching architectures
            preprocessor: Fitted feature preprocessor
            device: Device to use for prediction
            
        Returns:
            Predictor instance with loaded weights
        """
        # Create predictor
        predictor = cls(encoder, task_head, preprocessor, device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=predictor.device)
        
        # Load encoder
        predictor.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        
        # Load task heads
        for name, state_dict in checkpoint["task_heads_state_dict"].items():
            if name in predictor.task_heads:
                predictor.task_heads[name].load_state_dict(state_dict)
            else:
                predictor.logger.warning(
                    f"Task head '{name}' in checkpoint not found in current model"
                )
        
        return predictor
