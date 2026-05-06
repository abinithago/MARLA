"""
Optimizers for MARLA (Misclassification Aware Rank-Limited Adaptation).

Provides optimizer and scheduler factories for training MARLA models.
"""

import torch
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def sgd_optimizer(model, lr, momentum=0.0, weight_decay=0.0):
    """
    Create SGD optimizer for MARLA model.
    
    Args:
        model: MARLA model (or any PyTorch model)
        lr: Learning rate
        momentum: Momentum factor (default: 0.0)
        weight_decay: Weight decay (L2 regularization) coefficient (default: 0.0)
    
    Returns:
        optimizer: SGD optimizer instance
    """
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def sgd_optimizer_from_params(params, lr, momentum=0.0, weight_decay=0.0):
    """
    Create SGD optimizer from parameter list.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0.0)
        weight_decay: Weight decay (L2 regularization) coefficient (default: 0.0)
    
    Returns:
        optimizer: SGD optimizer instance
    """
    optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def adamw_optimizer(model, lr, weight_decay=0.0):
    """
    Create AdamW optimizer for MARLA model.
    
    Args:
        model: MARLA model (or any PyTorch model)
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization) coefficient (default: 0.0)
    
    Returns:
        optimizer: AdamW optimizer instance
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def cosine_lr_scheduler(optimizer, num_epochs, eta_min=None):
    """
    Create cosine annealing learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        num_epochs: Total number of training epochs (T_max)
        eta_min: Minimum learning rate (default: lr / 100)
    
    Returns:
        scheduler: CosineAnnealingLR scheduler instance
    """
    if eta_min is None:
        # Get initial learning rate from optimizer
        lr = optimizer.param_groups[0]['lr']
        eta_min = lr / 100.0
    return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)


def constant_lr_scheduler(*_):
    """
    Constant learning rate scheduler (no scheduling).
    
    Returns None, indicating no scheduler should be used.
    """
    return None
