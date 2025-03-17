"""
Configuration management module for Tabular Transformer.

This module provides utilities for managing model configuration,
hyperparameters, and other settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class TransformerConfig:
    """Configuration for the Transformer encoder."""
    embed_dim: int = 256
    input_dim: Optional[int] = None
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    variational: bool = True
    beta: float = 1.0  # KL divergence weight


@dataclass
class TaskConfig:
    """Configuration for a task head."""
    name: str
    type: str  # classification, regression, survival, etc.
    output_dim: int
    target_columns: List[str]
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    weight: float = 1.0  # Loss weight for this task


@dataclass
class ModelConfig:
    """Complete model configuration."""
    transformer: TransformerConfig
    tasks: Dict[str, TaskConfig]
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    categorical_embedding_dims: Dict[str, int] = field(default_factory=dict)
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "cuda"  # cuda or cpu

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ModelConfig":
        """Create a ModelConfig instance from a dictionary."""
        transformer_config = TransformerConfig(**config_dict.get("transformer", {}))
        
        task_configs = {}
        for task_name, task_dict in config_dict.get("tasks", {}).items():
            task_configs[task_name] = TaskConfig(name=task_name, **task_dict)
        
        return cls(
            transformer=transformer_config,
            tasks=task_configs,
            numeric_columns=config_dict.get("numeric_columns", []),
            categorical_columns=config_dict.get("categorical_columns", []),
            categorical_embedding_dims=config_dict.get("categorical_embedding_dims", {}),
            batch_size=config_dict.get("batch_size", 64),
            learning_rate=config_dict.get("learning_rate", 1e-4),
            weight_decay=config_dict.get("weight_decay", 1e-5),
            max_epochs=config_dict.get("max_epochs", 100),
            early_stopping_patience=config_dict.get("early_stopping_patience", 10),
            device=config_dict.get("device", "cuda"),
        )
