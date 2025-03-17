"""
Simulation utilities for tabular transformer.

This module provides functionality for Monte Carlo simulations to
quantify uncertainty in predictions from tabular transformer models.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm.auto import tqdm

from tabular_transformer.utils.logger import LoggerMixin
from tabular_transformer.models.transformer_encoder import TabularTransformer
from tabular_transformer.models.task_heads.base import BaseTaskHead
from tabular_transformer.data.dataset import TabularDataset
from tabular_transformer.inference.predict import Predictor


class UncertaintySimulator(LoggerMixin):
    """
    Simulator for uncertainty quantification.
    
    This class provides methods for running Monte Carlo simulations
    to estimate uncertainty in model predictions.
    """
    
    def __init__(
        self,
        predictor: Predictor,
        num_samples: int = 100,
    ):
        """
        Initialize simulator.
        
        Args:
            predictor: Predictor instance for generating predictions
            num_samples: Number of Monte Carlo samples to generate
        """
        self.predictor = predictor
        self.num_samples = num_samples
        
        # Ensure the encoder is variational
        self.check_variational()
    
    def check_variational(self) -> None:
        """
        Check if the encoder is variational.
        
        This ensures that the encoder supports Monte Carlo sampling
        for uncertainty quantification.
        
        Raises:
            ValueError: If the encoder is not variational
        """
        # Run a simple forward pass to check if output is a tuple
        encoder = self.predictor.encoder
        device = self.predictor.device
        
        # Create a small dummy input
        dummy_numeric = torch.zeros((1, max(1, len(self.predictor.preprocessor.numeric_columns))), device=device)
        dummy_numeric_mask = torch.zeros_like(dummy_numeric)
        dummy_categorical = torch.zeros((1, max(1, len(self.predictor.preprocessor.categorical_columns))), dtype=torch.long, device=device)
        dummy_categorical_mask = torch.zeros_like(dummy_categorical)
        
        with torch.no_grad():
            # Forward pass
            output = encoder(
                numeric_features=dummy_numeric,
                numeric_mask=dummy_numeric_mask,
                categorical_features=dummy_categorical,
                categorical_mask=dummy_categorical_mask,
            )
        
        # Check if output is a tuple (z, mu, logvar)
        if not isinstance(output, tuple) or len(output) != 3:
            self.logger.warning(
                "Encoder does not appear to be variational. "
                "Monte Carlo simulation may not provide meaningful uncertainty estimates."
            )
    
    def monte_carlo_simulate(
        self,
        data: Union[pd.DataFrame, torch.utils.data.DataLoader],
        task_names: Optional[List[str]] = None,
        batch_size: int = 64,
        progress_bar: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run Monte Carlo simulation for uncertainty quantification.
        
        Args:
            data: Input data (DataFrame or DataLoader)
            task_names: Optional list of task names to simulate (all by default)
            batch_size: Batch size for prediction (ignored if DataLoader is provided)
            progress_bar: Whether to display progress bar
            
        Returns:
            Dict mapping task names to simulation results
        """
        # Determine which tasks to simulate
        if task_names is None:
            task_names = list(self.predictor.task_heads.keys())
        
        # Set prediction components to their evaluation mode
        self.predictor.encoder.eval()
        for head in self.predictor.task_heads.values():
            head.eval()
        
        # Create DataLoader if needed
        if isinstance(data, pd.DataFrame):
            # Create TabularDataset from DataFrame
            dataset = TabularDataset(
                dataframe=data,
                numeric_columns=self.predictor.preprocessor.numeric_columns,
                categorical_columns=self.predictor.preprocessor.categorical_columns,
                preprocessor=self.predictor.preprocessor,
                fit_preprocessor=False,
            )
            data_loader = dataset.create_dataloader(
                batch_size=batch_size, shuffle=False, num_workers=0
            )
        else:
            data_loader = data
        
        # Get encoder and device
        encoder = self.predictor.encoder
        device = self.predictor.device
        
        # Calculate number of samples in dataset
        num_samples = len(data_loader.dataset)
        
        # Storage for simulation results
        results = {task_name: {} for task_name in task_names}
        
        # Run simulation
        sample_range = range(self.num_samples)
        if progress_bar:
            sample_range = tqdm(
                sample_range, desc=f"Running {self.num_samples} MC simulations"
            )
        
        for mc_sample in sample_range:
            # Storage for this simulation
            simulation_results = []
            
            with torch.no_grad():
                for batch in data_loader:
                    # Move inputs to device
                    numeric_features = batch["numeric_features"].to(device)
                    numeric_mask = batch["numeric_mask"].to(device)
                    categorical_features = batch["categorical_features"].to(device)
                    categorical_mask = batch["categorical_mask"].to(device)
                    
                    # Forward pass through encoder
                    encoder_output = encoder(
                        numeric_features=numeric_features,
                        numeric_mask=numeric_mask,
                        categorical_features=categorical_features,
                        categorical_mask=categorical_mask,
                    )
                    
                    # Use output for predictions
                    if isinstance(encoder_output, tuple):
                        # Variational case - use sampled latent
                        z, _, _ = encoder_output
                    else:
                        # Non-variational case - use encoded features directly
                        z = encoder_output
                    
                    # Generate predictions for each task
                    batch_results = {}
                    for task_name in task_names:
                        head = self.predictor.task_heads[task_name]
                        predictions = head.predict(z)
                        
                        # Move predictions to CPU
                        for key, value in predictions.items():
                            if isinstance(value, torch.Tensor):
                                predictions[key] = value.cpu().numpy()
                        
                        batch_results[task_name] = predictions
                    
                    simulation_results.append(batch_results)
            
            # Combine batch results for this simulation
            for task_name in task_names:
                for batch_idx, batch_result in enumerate(simulation_results):
                    task_result = batch_result[task_name]
                    
                    if batch_idx == 0:
                        # First batch - initialize storage for this task
                        if mc_sample == 0:
                            # First simulation - initialize with None for each key
                            for key in task_result:
                                results[task_name][key] = []
                        
                        # Add storage for this simulation
                        for key in task_result:
                            results[task_name][key].append([])
                    
                    # Add batch result to this simulation
                    for key in task_result:
                        results[task_name][key][-1].append(task_result[key])
        
        # Combine batches for each simulation
        for task_name in task_names:
            for key in results[task_name]:
                # For each simulation...
                for i, sim_batches in enumerate(results[task_name][key]):
                    if isinstance(sim_batches[0], np.ndarray):
                        # Concatenate arrays
                        results[task_name][key][i] = np.concatenate(sim_batches, axis=0)
                    else:
                        # Convert to list
                        results[task_name][key][i] = [item for batch in sim_batches for item in batch]
                
                # Stack simulations along a new first axis
                results[task_name][key] = np.stack(results[task_name][key], axis=0)
        
        return results
    
    def calculate_statistics(
        self,
        simulation_results: Dict[str, Dict[str, np.ndarray]],
        percentiles: List[float] = [2.5, 25, 50, 75, 97.5],
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Calculate statistics from Monte Carlo simulation results.
        
        Args:
            simulation_results: Results from monte_carlo_simulate
            percentiles: Percentiles to calculate
            
        Returns:
            Dict with statistics for each task and prediction component
        """
        stats = {}
        
        for task_name, task_results in simulation_results.items():
            stats[task_name] = {}
            
            for key, simulations in task_results.items():
                # Calculate statistics along the simulation axis (0)
                stats[task_name][key] = {
                    "mean": np.mean(simulations, axis=0),
                    "std": np.std(simulations, axis=0),
                    "percentiles": {
                        p: np.percentile(simulations, p, axis=0)
                        for p in percentiles
                    },
                }
        
        return stats
    
    def simulate_and_calculate_statistics(
        self,
        data: Union[pd.DataFrame, torch.utils.data.DataLoader],
        task_names: Optional[List[str]] = None,
        batch_size: int = 64,
        percentiles: List[float] = [2.5, 25, 50, 75, 97.5],
        progress_bar: bool = True,
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Run Monte Carlo simulation and calculate statistics in one go.
        
        Args:
            data: Input data (DataFrame or DataLoader)
            task_names: Optional list of task names to simulate (all by default)
            batch_size: Batch size for prediction (ignored if DataLoader is provided)
            percentiles: Percentiles to calculate
            progress_bar: Whether to display progress bar
            
        Returns:
            Dict with statistics for each task and prediction component
        """
        # Run simulation
        simulations = self.monte_carlo_simulate(
            data=data,
            task_names=task_names,
            batch_size=batch_size,
            progress_bar=progress_bar,
        )
        
        # Calculate statistics
        return self.calculate_statistics(
            simulation_results=simulations,
            percentiles=percentiles,
        )
    
    def get_prediction_intervals(
        self,
        data: Union[pd.DataFrame, torch.utils.data.DataLoader],
        task_name: str,
        prediction_key: str = "mean",
        interval_width: float = 0.95,
        batch_size: int = 64,
        progress_bar: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals for a specific task and prediction component.
        
        Args:
            data: Input data (DataFrame or DataLoader)
            task_name: Task name to generate intervals for
            prediction_key: Key in the task head's prediction dictionary to use
            interval_width: Width of prediction interval (0-1)
            batch_size: Batch size for prediction
            progress_bar: Whether to display progress bar
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound) arrays
        """
        # Check that task exists
        if task_name not in self.predictor.task_heads:
            raise ValueError(f"Task '{task_name}' not found in model")
        
        # Calculate percentiles
        lower_percentile = ((1 - interval_width) / 2) * 100
        upper_percentile = 100 - lower_percentile
        
        # Run simulation and calculate statistics
        stats = self.simulate_and_calculate_statistics(
            data=data,
            task_names=[task_name],
            batch_size=batch_size,
            percentiles=[lower_percentile, 50, upper_percentile],
            progress_bar=progress_bar,
        )
        
        # Extract results
        task_stats = stats[task_name]
        if prediction_key not in task_stats:
            raise ValueError(
                f"Prediction key '{prediction_key}' not found in simulation results "
                f"for task '{task_name}'. Available keys: {list(task_stats.keys())}"
            )
        
        # Get mean and percentiles
        mean = task_stats[prediction_key]["mean"]
        lower_bound = task_stats[prediction_key]["percentiles"][lower_percentile]
        upper_bound = task_stats[prediction_key]["percentiles"][upper_percentile]
        
        return mean, lower_bound, upper_bound
