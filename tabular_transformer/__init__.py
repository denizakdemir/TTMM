"""
Tabular Transformer package for processing tabular data with transformers.

This package provides a comprehensive implementation of a transformer-based
model for tabular data, supporting multiple tasks, handling missing values,
and providing uncertainty quantification via variational inference.
"""

__version__ = '0.1.0'

# Import public classes and functions to make them available at package level
from tabular_transformer.models.transformer_encoder import TabularTransformer
from tabular_transformer.data.dataset import TabularDataset
from tabular_transformer.training.trainer import Trainer
from tabular_transformer.inference.predict import Predictor
