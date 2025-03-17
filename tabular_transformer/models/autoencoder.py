"""
Autoencoder for tabular transformer.

This module implements an autoencoder that can be used with
the tabular transformer for unsupervised learning and reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from tabular_transformer.utils.logger import LoggerMixin
from tabular_transformer.models.transformer_encoder import TabularTransformer


class AutoEncoder(nn.Module, LoggerMixin):
    """
    Autoencoder for tabular data.
    
    This component can be used for unsupervised learning,
    dimensionality reduction, and feature learning with
    the tabular transformer encoder.
    """
    
    def __init__(
        self,
        encoder: TabularTransformer,
        decoder_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        """
        Initialize autoencoder.
        
        Args:
            encoder: Tabular transformer encoder
            decoder_dims: List of decoder hidden dimensions
            output_dim: Output dimension (typically equal to input dimension)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.encoder = encoder
        embed_dim = encoder.config.embed_dim
        
        # Build decoder network
        layers = []
        prev_dim = embed_dim
        
        for hidden_dim in decoder_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(
        self,
        numeric_features: torch.Tensor,
        numeric_mask: torch.Tensor,
        categorical_features: torch.Tensor,
        categorical_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            numeric_features: Numeric feature tensor [batch_size, numeric_dim]
            numeric_mask: Mask for missing numeric values [batch_size, numeric_dim]
            categorical_features: Categorical feature indices [batch_size, num_categorical]
            categorical_mask: Mask for missing categorical values [batch_size, num_categorical]
            
        Returns:
            Dict with encoded features, decoder output, KL divergence (if available)
        """
        # Pass through encoder
        encoder_output = self.encoder(
            numeric_features=numeric_features,
            numeric_mask=numeric_mask,
            categorical_features=categorical_features,
            categorical_mask=categorical_mask,
        )
        
        # Handle variational case
        if isinstance(encoder_output, tuple):
            z, mu, logvar = encoder_output
            # Pass through decoder
            reconstructed = self.decoder(z)
            return {
                "encoded": z,
                "reconstructed": reconstructed,
                "mu": mu,
                "logvar": logvar,
            }
        else:
            # Non-variational case
            # Pass through decoder
            reconstructed = self.decoder(encoder_output)
            return {
                "encoded": encoder_output,
                "reconstructed": reconstructed,
            }
    
    def compute_reconstruction_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            original: Original input features
            reconstructed: Reconstructed features from decoder
            mask: Optional mask for missing values
            reduction: Loss reduction method
            
        Returns:
            Reconstruction loss
        """
        # Compute mean squared error
        loss = F.mse_loss(reconstructed, original, reduction="none")
        
        if mask is not None:
            # Apply mask
            loss = loss * (1 - mask)
            
            # Apply reduction
            if reduction == "mean":
                # Sum and divide by number of observed values
                num_observed = (1 - mask).sum().clamp(min=1.0)
                return loss.sum() / num_observed
            elif reduction == "sum":
                return loss.sum()
            else:  # 'none'
                return loss
        else:
            # Apply reduction
            if reduction == "mean":
                return loss.mean()
            elif reduction == "sum":
                return loss.sum()
            else:  # 'none'
                return loss
    
    def compute_loss(
        self,
        numeric_features: torch.Tensor,
        numeric_mask: torch.Tensor,
        categorical_features: torch.Tensor,
        categorical_mask: torch.Tensor,
        combined_original: torch.Tensor,
        combined_mask: Optional[torch.Tensor] = None,
        kl_weight: float = 1.0,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total autoencoder loss.
        
        Args:
            numeric_features: Numeric feature tensor [batch_size, numeric_dim]
            numeric_mask: Mask for missing numeric values [batch_size, numeric_dim]
            categorical_features: Categorical feature indices [batch_size, num_categorical]
            categorical_mask: Mask for missing categorical values [batch_size, num_categorical]
            combined_original: Original combined features for reconstruction target
            combined_mask: Optional mask for missing values in combined features
            kl_weight: Weight for KL divergence loss (for variational case)
            reduction: Loss reduction method
            
        Returns:
            Dict with total loss and individual loss components
        """
        # Forward pass
        outputs = self.forward(
            numeric_features=numeric_features,
            numeric_mask=numeric_mask,
            categorical_features=categorical_features,
            categorical_mask=categorical_mask,
        )
        
        # Compute reconstruction loss
        recon_loss = self.compute_reconstruction_loss(
            original=combined_original,
            reconstructed=outputs["reconstructed"],
            mask=combined_mask,
            reduction=reduction,
        )
        
        # Check if variational
        if "mu" in outputs and "logvar" in outputs:
            # Compute KL divergence
            mu, logvar = outputs["mu"], outputs["logvar"]
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            if reduction == "mean":
                kl_loss = kl_loss / mu.size(0)
                
            # Combine losses
            total_loss = recon_loss + kl_weight * kl_loss
            
            return {
                "total_loss": total_loss,
                "reconstruction_loss": recon_loss,
                "kl_loss": kl_loss,
            }
        else:
            # Non-variational case - just return reconstruction loss
            return {
                "total_loss": recon_loss,
                "reconstruction_loss": recon_loss,
            }
