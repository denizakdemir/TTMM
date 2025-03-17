"""
Transformer encoder for tabular data.

This module implements a transformer encoder with variational inference
capabilities for tabular data processing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from tabular_transformer.utils.logger import LoggerMixin
from tabular_transformer.utils.config import TransformerConfig


class CategoricalEmbeddings(nn.Module):
    """
    Module to handle embeddings for categorical features.
    """
    
    def __init__(
        self,
        categorical_dims: Dict[str, int],
        embedding_dims: Dict[str, int],
        dropout: float = 0.1,
    ):
        """
        Initialize categorical embeddings.
        
        Args:
            categorical_dims: Dict mapping column names to number of categories
            embedding_dims: Dict mapping column names to embedding dimensions
            dropout: Dropout probability for embeddings
        """
        super().__init__()
        
        self.column_names = list(categorical_dims.keys())
        self.embedding_layers = nn.ModuleDict()
        self.column_to_safe_name = {}
        
        for col in self.column_names:
            num_categories = categorical_dims[col]
            embed_dim = embedding_dims[col]
            
            # Create a safe name for the ModuleDict by replacing dots with underscores
            safe_col_name = col.replace('.', '_')
            self.column_to_safe_name[col] = safe_col_name
            
            # Create embedding layer with additional index for missing values
            self.embedding_layers[safe_col_name] = nn.Embedding(
                num_embeddings=num_categories,
                embedding_dim=embed_dim,
                padding_idx=0,  # Use index 0 for padding
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, categorical_features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get embeddings for categorical features.
        
        Args:
            categorical_features: Tensor of categorical indices [batch_size, num_features]
            mask: Optional mask for missing values [batch_size, num_features]
            
        Returns:
            Tensor of concatenated embeddings [batch_size, total_embedding_dim]
        """
        # Check if we have any categorical features
        if not self.column_names:
            return torch.zeros(categorical_features.shape[0], 0, device=categorical_features.device)
        
        # Get individual embeddings for each column
        embedded_features = []
        
        for i, col in enumerate(self.column_names):
            # Get indices for this column
            indices = categorical_features[:, i]
            
            # Get embeddings using the safe column name
            safe_col_name = self.column_to_safe_name[col]
            embedding = self.embedding_layers[safe_col_name](indices)
            
            # Apply mask if provided
            if mask is not None:
                col_mask = mask[:, i].unsqueeze(1)
                embedding = embedding * (1 - col_mask)
            
            embedded_features.append(embedding)
        
        # Concatenate embeddings
        if embedded_features:
            concatenated = torch.cat(embedded_features, dim=1)
            return self.dropout(concatenated)
        else:
            return torch.zeros(categorical_features.shape[0], 0, device=categorical_features.device)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    This adds positional information to input embeddings.
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_length, embedding_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class VariationalLayer(nn.Module):
    """
    Variational layer for uncertainty estimation.
    
    This layer implements a variational latent variable using
    the reparameterization trick.
    """
    
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize variational layer.
        
        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Networks for mean and log variance
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to variational parameters and sample latent variable.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (sampled_latent, mean, log_variance)
        """
        # Calculate mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Sample using reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
    
    def sample(
        self, x: torch.Tensor, num_samples: int = 1
    ) -> List[torch.Tensor]:
        """
        Generate multiple samples from the variational distribution.
        
        Args:
            x: Input tensor
            num_samples: Number of samples to generate
            
        Returns:
            List of sampled latent variables
        """
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        
        samples = []
        for _ in range(num_samples):
            eps = torch.randn_like(std)
            z = mu + eps * std
            samples.append(z)
        
        return samples
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence between the variational distribution and a standard normal.
        
        Args:
            mu: Mean tensor
            logvar: Log variance tensor
            
        Returns:
            KL divergence tensor
        """
        # KL(N(mu, sigma) || N(0, 1))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl


class TabularTransformer(nn.Module, LoggerMixin):
    """
    Transformer model for tabular data.
    
    This model processes tabular data with both numeric and categorical features
    using a transformer architecture with optional variational inference.
    """
    
    def __init__(
        self,
        numeric_dim: int,
        categorical_dims: Dict[str, int],
        categorical_embedding_dims: Dict[str, int],
        config: TransformerConfig,
    ):
        """
        Initialize tabular transformer.
        
        Args:
            numeric_dim: Number of numeric features
            categorical_dims: Dict mapping categorical column names to number of categories
            categorical_embedding_dims: Dict mapping categorical column names to embedding dimensions
            config: Transformer configuration
        """
        super().__init__()
        
        # Save configuration
        self.config = config
        self.numeric_dim = numeric_dim
        self.categorical_dims = categorical_dims
        
        # Calculate dimensions
        total_cat_dim = sum(categorical_embedding_dims.values())
        feature_dim = numeric_dim + total_cat_dim
        
        # Feature processing components
        self.categorical_embeddings = CategoricalEmbeddings(
            categorical_dims=categorical_dims,
            embedding_dims=categorical_embedding_dims,
            dropout=config.dropout,
        )
        
        # Projection for numeric features
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_dim, numeric_dim),
            nn.LayerNorm(numeric_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        ) if numeric_dim > 0 else None
        
        # Feature projection to transformer dimension
        self.feature_projection = nn.Linear(feature_dim, config.embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=config.embed_dim,
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )
        
        # Optional variational layer
        self.variational = None
        if config.variational:
            self.variational = VariationalLayer(
                input_dim=config.embed_dim,
                latent_dim=config.embed_dim,
            )
    
    def forward(
        self,
        numeric_features: torch.Tensor,
        numeric_mask: torch.Tensor,
        categorical_features: torch.Tensor,
        categorical_mask: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            numeric_features: Numeric feature tensor [batch_size, numeric_dim]
            numeric_mask: Mask for missing numeric values [batch_size, numeric_dim]
            categorical_features: Categorical feature indices [batch_size, num_categorical]
            categorical_mask: Mask for missing categorical values [batch_size, num_categorical]
            
        Returns:
            If variational: Tuple of (encoded_features, mu, logvar)
            If not variational: encoded_features
        """
        batch_size = numeric_features.shape[0]
        
        # Process numeric features if present
        if self.numeric_dim > 0:
            numeric_features = numeric_features * (1 - numeric_mask)
            if self.numeric_projection is not None:
                numeric_features = self.numeric_projection(numeric_features)
        
        # Process categorical features
        categorical_embeddings = self.categorical_embeddings(
            categorical_features, categorical_mask
        )
        
        # Combine features
        if self.numeric_dim > 0:
            combined_features = torch.cat(
                [numeric_features, categorical_embeddings], dim=1
            )
        else:
            combined_features = categorical_embeddings
        
        # Project to transformer dimension
        projected_features = self.feature_projection(combined_features)
        
        # Add positional encoding (treating each feature as a sequence position)
        positioned_features = self.positional_encoding(
            projected_features.unsqueeze(1)
        ).squeeze(1)
        
        # Pass through transformer encoder
        # Note: We're using a "sequence length" of 1 here because we're processing
        # all features as a single token. For more complex sequence modeling,
        # you would reshape the features differently.
        encoded_features = self.transformer_encoder(
            positioned_features.unsqueeze(1)
        ).squeeze(1)
        
        # Apply variational layer if configured
        if self.variational is not None:
            z, mu, logvar = self.variational(encoded_features)
            return z, mu, logvar
        else:
            return encoded_features
    
    def compute_kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss for variational inference.
        
        Args:
            mu: Mean tensor
            logvar: Log variance tensor
            
        Returns:
            KL divergence loss
        """
        if self.variational is None:
            raise ValueError("Model is not configured for variational inference")
        
        return self.variational.kl_divergence(mu, logvar)
