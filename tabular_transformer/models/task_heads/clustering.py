"""
Clustering task head for tabular transformer.

This module implements a clustering head for the tabular transformer model,
enabling unsupervised learning with the transformer encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from tabular_transformer.models.task_heads.base import BaseTaskHead


class ClusteringHead(BaseTaskHead):
    """
    Clustering task head for tabular transformer.
    
    This head enables unsupervised clustering of tabular data by learning
    a mapping from the encoder output to cluster assignments.
    """
    
    def __init__(
        self,
        name: str,
        input_dim: int,
        num_clusters: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        distance_type: str = "cosine",
    ):
        """
        Initialize clustering head.
        
        Args:
            name: Name of the task
            input_dim: Input dimension from encoder
            num_clusters: Number of clusters
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            temperature: Temperature parameter for soft assignments
            distance_type: Type of distance metric ('cosine', 'euclidean')
        """
        super().__init__(
            name=name,
            input_dim=input_dim,
            output_dim=input_dim,  # We'll project back to input_dim for clustering
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.distance_type = distance_type
        
        # Create cluster centers
        self.cluster_centers = nn.Parameter(
            torch.randn(num_clusters, input_dim)
        )
        # Initialize cluster centers using Xavier initialization
        nn.init.xavier_uniform_(self.cluster_centers)
        
        # For reconstruction loss (optional)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the clustering head.
        
        Args:
            x: Input tensor from encoder [batch_size, input_dim]
            
        Returns:
            Projected features for clustering [batch_size, output_dim]
        """
        return self.network(x)
    
    def compute_distances(
        self, x: torch.Tensor, centers: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute distances between samples and cluster centers.
        
        Args:
            x: Input embeddings [batch_size, embed_dim]
            centers: Optional custom centers [num_clusters, embed_dim]
                Default is to use the learned cluster centers
            
        Returns:
            Distance tensor [batch_size, num_clusters]
        """
        if centers is None:
            centers = self.cluster_centers
        
        if self.distance_type == "cosine":
            # Normalize embeddings and centers
            x_norm = F.normalize(x, p=2, dim=1)
            centers_norm = F.normalize(centers, p=2, dim=1)
            
            # Compute cosine similarity
            similarity = torch.matmul(x_norm, centers_norm.t())
            
            # Convert similarity to distance (1 - similarity)
            distances = 1 - similarity
        else:  # euclidean
            # Compute squared Euclidean distances
            x_squared = torch.sum(x ** 2, dim=1, keepdim=True)
            centers_squared = torch.sum(centers ** 2, dim=1, keepdim=True).t()
            cross_term = torch.matmul(x, centers.t())
            
            distances = x_squared + centers_squared - 2 * cross_term
        
        return distances
    
    def compute_soft_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments.
        
        Args:
            x: Input embeddings [batch_size, embed_dim]
            
        Returns:
            Soft assignment probabilities [batch_size, num_clusters]
        """
        # Get distances
        distances = self.compute_distances(x)
        
        # Convert distances to probabilities using softmax with temperature
        # Lower temperature -> harder assignments
        return F.softmax(-distances / self.temperature, dim=1)
    
    def predict_clusters(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict cluster assignments.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Cluster indices [batch_size]
        """
        # Project features
        features = self.forward(x)
        
        # Compute distances to cluster centers
        distances = self.compute_distances(features)
        
        # Assign to closest cluster
        return torch.argmin(distances, dim=1)
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute clustering loss (unsupervised).
        
        Note: targets are not used for unsupervised clustering,
        but the parameter is kept for API consistency.
        
        Args:
            predictions: Projected features from model [batch_size, output_dim]
            targets: Not used (kept for API consistency)
            mask: Optional mask for missing data
            reduction: Loss reduction method
            
        Returns:
            Clustering loss (combination of clustering and reconstruction loss)
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Get soft assignments
        soft_assignments = self.compute_soft_assignments(predictions)
        
        # Compute target distribution for clustering
        # This follows Deep Clustering approach: t-distribution for soft assignments,
        # and squaring + normalizing for target distribution
        
        # Calculate auxiliary target distribution (KL divergence target)
        # p_ij = soft assignment of sample i to cluster j
        # q_ij = target distribution
        # Formula: q_ij = (p_ij^2 / sum_i(p_ij)) / (sum_j(p_ij^2 / sum_i(p_ij)))
        numerator = soft_assignments ** 2 / torch.sum(soft_assignments, dim=0)
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        target_distribution = numerator / denominator
        
        # Stop gradient for target distribution
        target_distribution = target_distribution.detach()
        
        # KL divergence loss
        # DKL(P||Q) = sum_i sum_j p_ij * log(p_ij / q_ij)
        kl_loss = F.kl_div(
            soft_assignments.log(), target_distribution, reduction="none"
        ).sum(dim=1)
        
        # Optional: Reconstruction loss for better feature learning
        reconstructed = self.decoder(predictions)
        
        # We don't have the original features here, so reconstruct to encoder output
        # In an actual implementation, you might want to reconstruct to original features
        # or pass them as additional arguments
        recon_loss = F.mse_loss(reconstructed, predictions, reduction="none").mean(dim=1)
        
        # Combine losses
        combined_loss = kl_loss + 0.1 * recon_loss
        
        # Apply mask if provided
        if mask is not None:
            combined_loss = combined_loss * mask
        
        # Apply reduction
        if reduction == "mean":
            # If mask provided, compute mean over valid samples
            if mask is not None:
                return combined_loss.sum() / mask.sum().clamp(min=1.0)
            else:
                return combined_loss.mean()
        elif reduction == "sum":
            return combined_loss.sum()
        else:  # 'none'
            return combined_loss
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate clustering predictions.
        
        Args:
            x: Input tensor from encoder
            
        Returns:
            Dict containing cluster assignments and probabilities
        """
        # Project features
        features = self.forward(x)
        
        # Get soft assignments
        soft_assignments = self.compute_soft_assignments(features)
        
        # Get hard cluster assignments
        cluster_ids = torch.argmax(soft_assignments, dim=1)
        
        return {
            "features": features,
            "cluster_probabilities": soft_assignments,
            "cluster_ids": cluster_ids,
        }
    
    def update_cluster_centers(self, features: torch.Tensor) -> None:
        """
        Update cluster centers based on current data.
        
        This is typically used during initialization or occasional updates.
        
        Args:
            features: Feature embeddings [num_samples, embed_dim]
        """
        # Use k-means++ initialization to set initial centers
        batch_size = features.shape[0]
        embed_dim = features.shape[1]
        device = features.device
        
        # Choose first center randomly
        first_idx = torch.randint(0, batch_size, (1,)).item()
        centers = [features[first_idx]]
        
        # Choose remaining centers based on distance-weighted probabilities
        for i in range(1, self.num_clusters):
            # Compute distances to existing centers
            dists = torch.zeros(batch_size, device=device)
            for c in centers:
                dist = torch.sum((features - c) ** 2, dim=1)
                dists = torch.min(dists, dist) if dists.any() else dist
            
            # Choose next center with probability proportional to distance
            probs = dists / dists.sum()
            next_idx = torch.multinomial(probs, 1).item()
            centers.append(features[next_idx])
        
        # Convert to tensor and update cluster centers
        new_centers = torch.stack(centers)
        self.cluster_centers.data = new_centers
        
    def evaluate(
        self,
        predictions: Union[Dict[str, torch.Tensor], pd.DataFrame],
        targets: Optional[Union[torch.Tensor, pd.Series, pd.DataFrame]] = None,
        metric: str = "silhouette",
    ) -> float:
        """
        Evaluate clustering quality.
        
        For unsupervised clustering, this evaluates internal metrics of cluster quality,
        not accuracy against ground truth (which is usually not available).
        
        Args:
            predictions: Dict of predictions from predict method or DataFrame from predict_dataframe
            targets: Not used for clustering evaluation (kept for API consistency)
            metric: Evaluation metric ('silhouette', 'davies_bouldin', 'calinski_harabasz')
            
        Returns:
            Performance score (lower is better, so some metrics are negated)
        """
        # Get cluster assignments and features
        if isinstance(predictions, pd.DataFrame):
            # Handle DataFrame format from predict_dataframe
            if 'cluster_ids_0' in predictions.columns:
                labels = predictions['cluster_ids_0'].values
            else:
                # Try to find features and compute clusters
                feature_cols = [col for col in predictions.columns if col.startswith('features_')]
                if feature_cols:
                    features = predictions[feature_cols].values
                    # Convert to tensor for computation
                    features_tensor = torch.tensor(features, dtype=torch.float32)
                    labels = self.predict_clusters(features_tensor).cpu().numpy()
                else:
                    # Fallback to first column assuming it has cluster IDs
                    labels = predictions.iloc[:, 0].values
        else:
            # Handle dict format from predict method
            if "cluster_ids" in predictions:
                labels = predictions["cluster_ids"].cpu().numpy()
            else:
                # Compute from features
                labels = self.predict_clusters(predictions["features"]).cpu().numpy()
        
        # Get features for metrics that need them
        if isinstance(predictions, pd.DataFrame):
            feature_cols = [col for col in predictions.columns if col.startswith('features_')]
            if feature_cols:
                features = predictions[feature_cols].values
            else:
                # For metrics that need features but we don't have them, we'll handle later
                features = None
        else:
            features = predictions["features"].cpu().numpy() if "features" in predictions else None
        
        # Ensure labels are a numpy array
        labels = np.array(labels)
        
        # Check if we have enough samples and clusters for evaluation
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            # Need at least 2 clusters for evaluation metrics
            return 0.0
        
        # Calculate metric
        if metric == "silhouette" or metric == "default":
            # Silhouette score (higher is better, range [-1, 1])
            # Need features for distance computation
            if features is not None:
                score = silhouette_score(features, labels)
                # Negate to follow convention that lower is better
                return -score
            else:
                # Can't compute silhouette without features
                return 0.0
        
        elif metric == "davies_bouldin":
            # Davies-Bouldin index (lower is better)
            # Measures average similarity between clusters
            if features is not None:
                return davies_bouldin_score(features, labels)
            else:
                return 0.0
        
        elif metric == "calinski_harabasz":
            # Calinski-Harabasz index (higher is better)
            # Measures ratio of between-cluster to within-cluster variance
            if features is not None:
                score = calinski_harabasz_score(features, labels)
                # Negate to follow convention that lower is better
                return -score
            else:
                return 0.0
        
        elif metric == "cluster_stability":
            # Placeholder for custom cluster stability metric
            # This could compare cluster assignments across multiple runs or data subsets
            # Not implemented in this version
            return 0.0
        
        else:
            raise ValueError(f"Unknown metric for clustering: {metric}")
