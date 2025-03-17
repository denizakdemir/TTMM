"""
Data preprocessing utilities for tabular data.

This module provides functions for handling numeric and categorical features,
preprocessing missing values, and other data transformations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from tabular_transformer.utils.logger import LoggerMixin


class FeaturePreprocessor(LoggerMixin):
    """
    Preprocessor for tabular features.
    
    This class handles the preprocessing of numeric and categorical features,
    including standardization, missing value handling, and encoding.
    """
    
    def __init__(
        self,
        numeric_columns: List[str],
        categorical_columns: List[str],
        categorical_embedding_dims: Optional[Dict[str, int]] = None,
        standardize_numeric: bool = True,
    ):
        """
        Initialize the feature preprocessor.
        
        Args:
            numeric_columns: List of numeric column names
            categorical_columns: List of categorical column names
            categorical_embedding_dims: Dict mapping categorical column names to embedding dimensions
            standardize_numeric: Whether to standardize numeric features
        """
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.categorical_embedding_dims = categorical_embedding_dims or {}
        self.standardize_numeric = standardize_numeric
        
        # Initialize preprocessing components
        self.numeric_scaler = StandardScaler() if standardize_numeric else None
        self.categorical_encoders = {
            col: LabelEncoder() for col in categorical_columns
        }
        self.categorical_nunique = {}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        """
        Fit the preprocessor to the data.
        
        Args:
            df: Input DataFrame containing features
            
        Returns:
            Self
        """
        # Process numeric features
        if self.numeric_columns and self.standardize_numeric:
            # Replace NaNs with 0 for fitting the scaler
            numeric_data = df[self.numeric_columns].fillna(0)
            self.numeric_scaler.fit(numeric_data)
            self.logger.info(f"Fitted numeric scaler to {len(self.numeric_columns)} columns")
        
        # Process categorical features
        for col in self.categorical_columns:
            # Add a special "missing" category for NaN values
            series = df[col].fillna("__missing__")
            self.categorical_encoders[col].fit(series)
            n_categories = len(self.categorical_encoders[col].classes_)
            self.categorical_nunique[col] = n_categories
            
            # Auto-determine embedding dimension if not provided
            if col not in self.categorical_embedding_dims:
                # Common heuristic: min(50, (n_categories+1)//2)
                self.categorical_embedding_dims[col] = min(50, (n_categories + 1) // 2)
                
            self.logger.info(
                f"Column {col}: {n_categories} categories, "
                f"embedding dim {self.categorical_embedding_dims[col]}"
            )
        
        self.is_fitted = True
        return self
    
    def transform_numeric(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform numeric features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (transformed_numeric_array, missing_mask)
        """
        if not self.numeric_columns:
            return np.zeros((len(df), 0)), np.zeros((len(df), 0))
        
        # Extract numeric data
        numeric_data = df[self.numeric_columns].copy()
        
        # Create missing mask
        missing_mask = numeric_data.isna().values.astype(np.float32)
        
        # Fill missing values with 0 (will be ignored in calculations due to the mask)
        numeric_data = numeric_data.fillna(0)
        
        # Apply standardization if needed
        if self.standardize_numeric:
            numeric_data = self.numeric_scaler.transform(numeric_data)
        else:
            numeric_data = numeric_data.values
            
        return numeric_data.astype(np.float32), missing_mask
    
    def transform_categorical(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (transformed_categorical_array, missing_mask)
        """
        if not self.categorical_columns:
            return np.zeros((len(df), 0), dtype=np.int64), np.zeros((len(df), 0))
        
        categorical_data = np.zeros((len(df), len(self.categorical_columns)), dtype=np.int64)
        missing_mask = np.zeros((len(df), len(self.categorical_columns)), dtype=np.float32)
        
        for i, col in enumerate(self.categorical_columns):
            series = df[col].copy()
            
            # Identify missing values
            is_missing = series.isna()
            missing_mask[:, i] = is_missing.values.astype(np.float32)
            
            # Replace missing with placeholder and encode
            series = series.fillna("__missing__")
            try:
                categorical_data[:, i] = self.categorical_encoders[col].transform(series)
            except ValueError as e:
                # Handle unseen categories by setting them to a default value (0)
                self.logger.warning(f"Encountered unknown categories in {col}: {e}")
                categorical_data[:, i] = 0
        
        return categorical_data, missing_mask
    
    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform both numeric and categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (numeric_array, numeric_mask, categorical_array, categorical_mask)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        numeric_data, numeric_mask = self.transform_numeric(df)
        categorical_data, categorical_mask = self.transform_categorical(df)
        
        return numeric_data, numeric_mask, categorical_data, categorical_mask
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get the dimensions of features after preprocessing.
        
        Returns:
            Dict with feature dimensions
        """
        return {
            "numeric_dim": len(self.numeric_columns),
            "categorical_dims": {col: self.categorical_nunique[col] for col in self.categorical_columns},
            "categorical_embedding_dims": self.categorical_embedding_dims,
        }

    def get_total_embedding_dim(self) -> int:
        """
        Get the total dimension of all features after embedding.
        
        Returns:
            Total embedding dimension
        """
        numeric_dim = len(self.numeric_columns)
        categorical_dim = sum(self.categorical_embedding_dims.values())
        return numeric_dim + categorical_dim
