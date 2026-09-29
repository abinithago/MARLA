"""
Loss functions for MARLA (Misclassification Aware Rank-Limited Adaptation) and ERM (Empirical Risk Minimization).

- MARLA uses weighted cross-entropy loss with error-based reweighting and optional
  regularization on the adjustment matrix.
- ERM uses standard cross-entropy loss without any reweighting.
"""

import torch
import torch.nn.functional as F


def weighted_cross_entropy(logits, y, weights):
    """
    Weighted cross-entropy loss function.
    
    Args:
        logits: Tensor of shape (n, K) where K is number of classes
        y: Tensor of shape (n,) with class labels
        weights: Tensor of shape (n,) with per-example weights
    
    Returns:
        loss: Scalar weighted cross-entropy loss
    """
    # Clamp logits to prevent numerical instability
    logits = torch.clamp(logits, min=-100, max=100)
    ce = F.cross_entropy(logits, y, reduction='none')
    # Check for NaN or Inf in cross-entropy
    if torch.isnan(ce).any() or torch.isinf(ce).any():
        # Replace NaN/Inf with a large finite value
        ce = torch.where(torch.isnan(ce) | torch.isinf(ce), 
                         torch.tensor(100.0, device=ce.device, dtype=ce.dtype), ce)
    out = weights * ce
    return out.sum()


def get_marla_weights(logits, y, gamma):
    """
    Compute MARLA weights based on prediction confidence.
    
    Weight formula: μ_i = exp(γ * (1 - p_i))
    where p_i is the predicted probability of the correct label.
    
    Higher weight is assigned to examples with lower confidence (higher error),
    focusing the algorithm on hard examples.
    
    Args:
        logits: Tensor of shape (n, K) where K is number of classes
        y: Tensor of shape (n,) with class labels
        gamma: Reweighting strength parameter (gamma > 0)
    
    Returns:
        weights: Tensor of shape (n,) with normalized weights
    """
    p = logits.softmax(-1)
    y_onehot = torch.zeros_like(logits).scatter_(-1, y.unsqueeze(-1), 1)
    p_true = (p * y_onehot).sum(-1)  # Predicted probability of correct label
    
    # Weight: μ_i = exp(γ * (1 - p_i))
    # Higher weight for examples with lower confidence (lower p_i)
    # Use log-sum-exp trick for numerical stability
    log_weights = gamma * (1 - p_true)
    log_weights = log_weights - log_weights.max()
    weights = log_weights.exp()
    weights = weights / weights.sum()
    
    # Check for NaN or Inf in weights
    if torch.isnan(weights).any() or torch.isinf(weights).any():
        # Fallback to uniform weights if numerical issues
        weights = torch.ones_like(weights) / weights.shape[0]
    
    # Detach weights to prevent gradient flow through weight computation
    weights = weights.detach()
    
    return weights


def compute_spectral_decomposition(embeddings, weights, k):
    """
    Compute spectral decomposition for MARLA.
    
    Computes top-k eigenvectors of error-weighted covariance matrix.
    This identifies the k-dimensional subspace where most error variance occurs.
    
    Args:
        embeddings: Tensor of shape (n, d) where d is embedding dimension
        weights: Tensor of shape (n,) with per-example weights
        k: Number of top eigenvectors to extract (spectral rank)
    
    Returns:
        V_k: Top-k eigenvectors, shape (d, k)
        eigenvalues: Top-k eigenvalues, shape (k,)
    """
    # Normalize weights
    weights = weights / weights.sum()
    
    # Compute weighted mean
    mean_embedding = (weights.unsqueeze(-1) * embeddings).sum(dim=0)
    
    # Compute error-weighted covariance matrix
    centered = embeddings - mean_embedding.unsqueeze(0)
    covariance = (weights.unsqueeze(-1) * centered).T @ centered
    
    # Compute top-k eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    # Sort in descending order
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Return top-k
    V_k = eigenvectors[:, :k]
    top_eigenvalues = eigenvalues[:k]
    
    return V_k, top_eigenvalues


def marla_loss(logits, y, weights, reg_coeff=0.0, adjustment_matrix=None):
    """
    MARLA (Misclassification Aware Rank-Limited Adaptation) loss function.
    
    Uses weighted cross-entropy with optional regularization on adjustment matrix.
    
    Args:
        logits: Tensor of shape (n, K) with model predictions
        y: Tensor of shape (n,) with class labels
        weights: Tensor of shape (n,) with per-example weights
        reg_coeff: Regularization coefficient for adjustment matrix (default: 0.0)
        adjustment_matrix: Adjustment matrix to regularize, shape (k, C) (optional)
    
    Returns:
        loss: Scalar loss value
    """
    loss = weighted_cross_entropy(logits, y, weights)
    
    # Add regularization on adjustment matrix if provided
    if adjustment_matrix is not None and reg_coeff > 0:
        reg = reg_coeff * (adjustment_matrix.pow(2).sum())
        loss = loss + reg
    return loss


def marla(*_):
    """
    MARLA loss function factory.
    
    Returns a criterion function that can be used with training loops.
    """
    def criterion(logits, y, weights, reg_coeff=0.0, adjustment_matrix=None):
        return marla_loss(logits, y, weights, reg_coeff, adjustment_matrix)
    return criterion


def erm_loss(logits, y):
    """
    ERM (Empirical Risk Minimization) loss function.
    
    Uses standard cross-entropy loss without any reweighting or special handling.
    This is the baseline method that minimizes average training loss.
    
    Args:
        logits: Tensor of shape (n, K) with model predictions
        y: Tensor of shape (n,) with class labels
    
    Returns:
        loss: Scalar cross-entropy loss
    """
    # Clamp logits to prevent numerical instability
    logits = torch.clamp(logits, min=-100, max=100)
    loss = F.cross_entropy(logits, y, reduction='mean')
    
    # Check for NaN or Inf
    if torch.isnan(loss) or torch.isinf(loss):
        # Fallback: compute per-example loss and handle NaN/Inf
        ce = F.cross_entropy(logits, y, reduction='none')
        ce = torch.where(torch.isnan(ce) | torch.isinf(ce),
                         torch.tensor(100.0, device=ce.device, dtype=ce.dtype), ce)
        loss = ce.mean()
    
    return loss


def erm(*_):
    """
    ERM loss function factory.

    Returns a criterion function that takes (logits, y, counter) and computes
    standard cross-entropy loss (ignoring counter).
    """
    def criterion(logits, y, counter=None):
        return torch.nn.functional.cross_entropy(logits, y)
    return criterion
