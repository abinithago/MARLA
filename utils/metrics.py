"""
Metrics computation for MARLA training and evaluation.

Computes mean accuracy, class balanced accuracy, worst group accuracy (WGA),
and worst 10th percentile accuracy.
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_mean_accuracy(logits, labels):
    """
    Compute mean accuracy (overall accuracy across all examples).
    
    Args:
        logits: Model logits, shape (n, C)
        labels: True labels, shape (n,)
    
    Returns:
        mean_acc: Scalar mean accuracy
    """
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float()
    mean_acc = correct.mean().item()
    return mean_acc


def compute_class_balanced_accuracy(logits, labels):
    """
    Compute class-balanced accuracy (average of per-class accuracies).
    
    This metric handles class imbalance by giving equal weight to each class,
    regardless of class frequency.
    
    Args:
        logits: Model logits, shape (n, C)
        labels: True labels, shape (n,)
    
    Returns:
        balanced_acc: Scalar class-balanced accuracy
        per_class_accs: Dictionary of {class_id: accuracy}
    """
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float()
    
    unique_classes = torch.unique(labels)
    per_class_accs = {}
    class_accs = []
    
    for class_id in unique_classes:
        mask = labels == class_id
        if mask.sum() > 0:
            class_acc = correct[mask].mean().item()
            per_class_accs[class_id.item()] = class_acc
            class_accs.append(class_acc)
        else:
            per_class_accs[class_id.item()] = 0.0
            class_accs.append(0.0)
    
    balanced_acc = np.mean(class_accs) if class_accs else 0.0
    
    return balanced_acc, per_class_accs


def compute_worst_group_accuracy(logits, labels, groups):
    """
    Compute worst group accuracy (WGA) - minimum accuracy across groups.
    
    Args:
        logits: Model logits, shape (n, C)
        labels: True labels, shape (n,)
        groups: Group assignments, shape (n,). Groups with value -1 are ignored (masked).
    
    Returns:
        wga: Scalar worst group accuracy
        group_accs: Dictionary of {group_id: accuracy}
    """
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float()
    
    unique_groups = torch.unique(groups)
    group_accs = {}
    
    for group_id in unique_groups:
        # Skip masked groups (value -1)
        if group_id.item() == -1:
            continue
        mask = groups == group_id
        if mask.sum() > 0:
            group_acc = correct[mask].mean().item()
            group_accs[group_id.item()] = group_acc
        else:
            group_accs[group_id.item()] = 0.0
    
    wga = min(group_accs.values()) if group_accs else 0.0
    
    return wga, group_accs


def compute_percentile_accuracy(logits, labels, percentile=10):
    """
    Compute percentile accuracy - accuracy on examples at a specific percentile by loss.
    
    This metric identifies examples at a specific percentile (by cross-entropy loss)
    and computes accuracy on those examples.
    
    Args:
        logits: Model logits, shape (n, C)
        labels: True labels, shape (n,)
        percentile: Percentile to use (e.g., 10 for 10th percentile, 50 for 50th percentile)
                    Lower percentile = harder examples (higher loss)
    
    Returns:
        pct_acc: Scalar accuracy at the specified percentile
        n_examples: Number of examples at this percentile
        losses: Per-example losses, shape (n,)
    """
    # Compute per-example cross-entropy losses
    losses = F.cross_entropy(logits, labels, reduction='none')
    
    # Get examples at specified percentile
    threshold = torch.quantile(losses, 1 - percentile/100)
    mask = losses >= threshold
    
    # Compute accuracy on these examples
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float()
    
    if mask.sum() > 0:
        pct_acc = correct[mask].mean().item()
        n_examples = mask.sum().item()
    else:
        pct_acc = 0.0
        n_examples = 0
    
    return pct_acc, n_examples, losses


def compute_worst_percentile_accuracy(logits, labels, percentile=10):
    """
    Compute worst 10th percentile accuracy - accuracy on the worst examples by loss.
    
    This metric identifies the top percentile of hardest examples (by cross-entropy loss)
    and computes accuracy on those examples. This approximates worst-group performance
    without requiring group labels.
    
    Args:
        logits: Model logits, shape (n, C)
        labels: True labels, shape (n,)
        percentile: Percentile to use (default: 10, meaning top 10% hardest)
    
    Returns:
        worst_pct_acc: Scalar accuracy on hardest percentile
        n_hard: Number of examples in the hardest percentile
        losses: Per-example losses, shape (n,)
    """
    return compute_percentile_accuracy(logits, labels, percentile=percentile)


def compute_logit_entropy(logits, labels):
    """
    Compute logit entropy - entropy of the predicted probability distribution.
    
    Higher entropy indicates more uncertainty in predictions.
    This can be used as a label-free metric to identify hard examples.
    
    Args:
        logits: Model logits, shape (n, C)
        labels: True labels, shape (n,)
    
    Returns:
        mean_entropy: Mean entropy across all examples
        entropy_per_example: Entropy per example, shape (n,)
    """
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Compute entropy: -sum(p * log(p))
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    mean_entropy = entropy.mean().item()
    
    return mean_entropy, entropy


def compute_exponentially_weighted_validation_metric(logits, labels, exp_weight_factor=1.0):
    """
    Compute exponentially weighted validation metric.
    
    This metric applies exponential weighting to validation examples based on their loss,
    giving higher weight to examples with higher loss. This is used for model selection
    (not for training).
    
    Args:
        logits: Model logits, shape (n, C)
        labels: True labels, shape (n,)
        exp_weight_factor: Exponential weighting factor (default: 1.0)
    
    Returns:
        weighted_accuracy: Accuracy weighted by exp(exp_weight_factor * loss)
    """
    # Compute per-example cross-entropy losses
    losses = F.cross_entropy(logits, labels, reduction='none')
    
    # Compute exponential weights: w_i = exp(exp_weight_factor * loss_i)
    # Use log-sum-exp trick for numerical stability
    log_weights = exp_weight_factor * losses
    log_weights = log_weights - log_weights.max()
    weights = log_weights.exp()
    weights = weights / weights.sum()  # Normalize
    
    # Compute accuracy
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float()
    
    # Weighted accuracy
    weighted_accuracy = (weights * correct).sum().item()
    
    return weighted_accuracy


def compute_all_metrics(logits, labels, groups=None, exp_weight_factor=None):
    """
    Compute all metrics: mean accuracy, class balanced accuracy, WGA, and exponentially weighted accuracy.
    
    Args:
        logits: Model logits, shape (n, C)
        labels: True labels, shape (n,)
        groups: Group assignments, shape (n,) (optional, required for WGA)
        exp_weight_factor: If provided, compute exponentially weighted validation metric (default: None)
    
    Returns:
        Dictionary with all computed metrics:
            - mean_accuracy: Overall mean accuracy
            - class_balanced_accuracy: Class-balanced accuracy
            - worst_group_accuracy: WGA (if groups provided, else None)
            - per_class_accuracies: Dict of per-class accuracies
            - group_accuracies: Dict of per-group accuracies (if groups provided)
            - exp_weighted_accuracy: Exponentially weighted accuracy (if exp_weight_factor provided)
    """
    results = {}
    
    # Mean accuracy
    results['mean_accuracy'] = compute_mean_accuracy(logits, labels)
    
    # Class balanced accuracy
    balanced_acc, per_class_accs = compute_class_balanced_accuracy(logits, labels)
    results['class_balanced_accuracy'] = balanced_acc
    results['per_class_accuracies'] = per_class_accs
    
    # Worst group accuracy (if groups provided)
    if groups is not None:
        wga, group_accs = compute_worst_group_accuracy(logits, labels, groups)
        results['worst_group_accuracy'] = wga
        results['group_accuracies'] = group_accs
    else:
        results['worst_group_accuracy'] = None
        results['group_accuracies'] = None
    
    
    # Exponentially weighted accuracy (for validation selection only)
    if exp_weight_factor is not None:
        exp_weighted_acc = compute_exponentially_weighted_validation_metric(logits, labels, exp_weight_factor)
        results['exp_weighted_accuracy'] = exp_weighted_acc
    
    return results
