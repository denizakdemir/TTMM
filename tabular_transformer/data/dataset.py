"""
Dataset classes for tabular data.

This module provides dataset implementations for handling tabular data
with both numeric and categorical features.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import LabelEncoder

from tabular_transformer.data.preprocess import FeaturePreprocessor
from tabular_transformer.utils.logger import LoggerMixin


class TabularDataset(Dataset, LoggerMixin):
    """
    PyTorch Dataset for tabular data.
    
    This dataset handles both numeric and categorical features,
    creates missing value masks, and prepares data for the tabular transformer.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
        target_columns: Optional[Dict[str, List[str]]] = None,
        preprocessor: Optional[FeaturePreprocessor] = None,
        fit_preprocessor: bool = True,
    ):
        """
        Initialize the tabular dataset.
        
        Args:
            dataframe: Input DataFrame
            numeric_columns: List of numeric column names
            categorical_columns: List of categorical column names
            target_columns: Dict mapping task names to lists of target column names
            preprocessor: Optional pre-fitted FeaturePreprocessor
            fit_preprocessor: Whether to fit the preprocessor (if not provided)
        """
        self.data = dataframe.copy()
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.target_columns = target_columns or {}
        
        # Initialize or use provided preprocessor
        if preprocessor is None:
            self.preprocessor = FeaturePreprocessor(
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
            )
            if fit_preprocessor:
                self.preprocessor.fit(self.data)
        else:
            self.preprocessor = preprocessor
        
        # Transform features
        (
            self.numeric_features,
            self.numeric_missing_mask,
            self.categorical_features,
            self.categorical_missing_mask,
        ) = self.preprocessor.transform(self.data)
        
        # Process targets
        self.targets = {}
        self.target_missing_masks = {}
        self.target_encoders = {}
        
        for task_name, cols in self.target_columns.items():
            if cols:
                target_df = self.data[cols]
                self.target_missing_masks[task_name] = target_df.isna().values.astype(np.float32)
                
                # Check if the target values are strings and need encoding
                # This is particularly important for classification tasks with string labels
                if target_df.dtypes.iloc[0] == 'object':
                    self.logger.info(f"Detected string values in target column '{cols[0]}', auto-encoding to numeric")
                    encoder = LabelEncoder()
                    # Fill missing with placeholder string for encoding
                    encoded_series = encoder.fit_transform(target_df.fillna('__missing__').iloc[:, 0])
                    # Store the encoder for potential inverse transform later
                    self.target_encoders[task_name] = encoder
                    self.targets[task_name] = encoded_series.reshape(-1, 1).astype(np.float32)
                else:
                    # Fill missing with zeros for now, they'll be ignored in loss computation
                    self.targets[task_name] = target_df.fillna(0).values.astype(np.float32)
        
        self.logger.info(
            f"Created dataset with {len(self.data)} samples, "
            f"{len(self.numeric_columns)} numeric features, "
            f"{len(self.categorical_columns)} categorical features, "
            f"{len(self.target_columns)} tasks"
        )
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing features, masks, and targets
        """
        sample = {
            "numeric_features": torch.tensor(self.numeric_features[idx], dtype=torch.float32),
            "numeric_mask": torch.tensor(self.numeric_missing_mask[idx], dtype=torch.float32),
            "categorical_features": torch.tensor(self.categorical_features[idx], dtype=torch.long),
            "categorical_mask": torch.tensor(self.categorical_missing_mask[idx], dtype=torch.float32),
        }
        
        # Add targets if available
        for task_name, target_array in self.targets.items():
            sample[f"target_{task_name}"] = torch.tensor(target_array[idx], dtype=torch.float32)
            sample[f"target_mask_{task_name}"] = torch.tensor(
                self.target_missing_masks[task_name][idx], dtype=torch.float32
            )
        
        return sample

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
        target_columns: Dict[str, List[str]],
        validation_split: Optional[float] = None,
        test_split: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple["TabularDataset", Optional["TabularDataset"], Optional["TabularDataset"]]:
        """
        Create datasets from a DataFrame with optional train/val/test splits.
        
        Args:
            dataframe: Input DataFrame
            numeric_columns: List of numeric column names
            categorical_columns: List of categorical column names
            target_columns: Dict mapping task names to lists of target column names
            validation_split: Optional fraction of data to use for validation
            test_split: Optional fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            val_dataset and test_dataset may be None if splits are not requested
        """
        # Create preprocessor and fit on the entire dataset
        preprocessor = FeaturePreprocessor(
            numeric_columns=numeric_columns, 
            categorical_columns=categorical_columns,
        )
        preprocessor.fit(dataframe)
        
        # If no splits requested, return single dataset
        if validation_split is None and test_split is None:
            return (
                cls(dataframe, numeric_columns, categorical_columns, target_columns, preprocessor),
                None,
                None,
            )
        
        # Create train/val/test splits
        test_size = test_split or 0.0
        val_size = validation_split or 0.0
        
        if random_state is not None:
            np.random.seed(random_state)
        
        indices = np.random.permutation(len(dataframe))
        test_end = int(len(indices) * test_size) if test_size > 0 else 0
        val_end = test_end + int(len(indices) * val_size) if val_size > 0 else test_end
        
        test_indices = indices[:test_end] if test_size > 0 else None
        val_indices = indices[test_end:val_end] if val_size > 0 else None
        train_indices = indices[val_end:]
        
        train_df = dataframe.iloc[train_indices].reset_index(drop=True)
        val_df = dataframe.iloc[val_indices].reset_index(drop=True) if val_indices is not None else None
        test_df = dataframe.iloc[test_indices].reset_index(drop=True) if test_indices is not None else None
        
        # Create datasets
        train_dataset = cls(
            train_df, numeric_columns, categorical_columns, target_columns, preprocessor
        )
        
        val_dataset = None
        if val_df is not None:
            val_dataset = cls(
                val_df, numeric_columns, categorical_columns, target_columns, preprocessor, fit_preprocessor=False
            )
        
        test_dataset = None
        if test_df is not None:
            test_dataset = cls(
                test_df, numeric_columns, categorical_columns, target_columns, preprocessor, fit_preprocessor=False
            )
        
        return train_dataset, val_dataset, test_dataset

    def create_dataloader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 0
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for loading
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )
