"""
MARLA (Misclassification Aware Rank-Limited Adaptation) model.

Provides MARLAModel class that adds a low-rank adjustment to a base classifier.
"""

from .architectures import get_model, get_classifier_and_feature_extractor
import torch
import torch.nn as nn
import torch.nn.functional as F


class MARLAModel(nn.Module):
    """
    MARLA (Misclassification Aware Rank-Limited Adaptation) Model.
    
    Adjusts base classifier logits by learning a low-rank adjustment in the
    error subspace identified by spectral decomposition.
    
    Forward pass:
        base_logits = W φ(x) + b                    # Base classifier
        projected = V_k^T φ(x)                      # Project to error subspace
        adjustment_logits = A^T projected           # Apply learned adjustment
        adjusted_logits = base_logits + adjustment_logits  # Combine
    """
    def __init__(self, base_weight, base_bias, V_k):
        """
        Args:
            base_weight: Tensor of shape (C, d) - base classifier weights (frozen)
            base_bias: Tensor of shape (C,) - base classifier bias (frozen)
            V_k: Tensor of shape (d, k) - error subspace basis (frozen)
        """
        super().__init__()
        self.base_weight = base_weight  # Frozen
        self.base_bias = base_bias      # Frozen
        self.V_k = V_k                  # Frozen
        
        # Learnable adjustment matrix: A ∈ ℝ^(k×C)
        # Initialized to zeros so initial output is just base classifier
        k, num_classes = V_k.shape[1], base_weight.shape[0]
        self.adjustment = nn.Parameter(torch.zeros(k, num_classes))
    
    def forward(self, embeddings):
        """
        Apply error subspace adjustment to classifier logits.
        
        Args:
            embeddings: Tensor of shape (batch_size, d) - input embeddings
        
        Returns:
            adjusted_logits: Tensor of shape (batch_size, C) - adjusted logits
        """
        base_logits = F.linear(embeddings, self.base_weight, self.base_bias)
        projected = F.linear(embeddings, self.V_k.t(), None)
        adjustment_logits = F.linear(projected, self.adjustment.t(), None)
        adjusted_logits = base_logits + adjustment_logits
        
        return adjusted_logits


__all__ = ['MARLAModel', 'get_model', 'get_classifier_and_feature_extractor']
