"""
Training utilities for Stage 1 ERM training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from .metrics import compute_all_metrics


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: callable,
    device: torch.device,
    epoch: int = 0
) -> Tuple[AverageMeter, Dict]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        epoch: Current epoch number
    
    Returns:
        loss_meter: AverageMeter with loss statistics
        acc_groups: Dictionary of {group_id: AverageMeter} with accuracy per group
    """
    model.train()
    loss_meter = AverageMeter()
    
    # Initialize accuracy meters per group
    acc_groups = {}
    
    for batch_idx, batch in enumerate(train_loader):
        # Handle different batch formats
        if len(batch) >= 3:
            x, y, *rest = batch
            groups = rest[0] if len(rest) > 0 else None
        else:
            x, y = batch
            groups = None
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(x)
        
        # Compute loss
        if isinstance(criterion, torch.nn.Module):
            loss = criterion(logits, y)
        else:
            loss = criterion(logits, y, epoch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update loss meter
        loss_meter.update(loss.item(), x.size(0))
        
        # Update accuracy per group
        predictions = torch.argmax(logits.detach(), dim=1)
        correct = (predictions == y).float()
        
        if groups is not None:
            groups = groups.to(device)
            unique_groups = torch.unique(groups)
            for g in unique_groups:
                if g.item() not in acc_groups:
                    acc_groups[g.item()] = AverageMeter()
                mask = groups == g
                if mask.sum() > 0:
                    acc = correct[mask].mean().item()
                    acc_groups[g.item()].update(acc, mask.sum().item())
    
    return loss_meter, acc_groups


def eval_model(
    model: torch.nn.Module,
    holdout_loaders: Dict[str, DataLoader],
    device: torch.device,
    is_bert: bool = False
) -> Dict[str, Dict]:
    """
    Evaluate model on validation and test sets.
    
    Args:
        model: Model to evaluate
        holdout_loaders: Dictionary with 'val' and 'test' loaders
        device: Device to run on
        is_bert: Whether this is a BERT model (handles input differently)
    
    Returns:
        Dictionary with results for each split. Each entry contains:
        - 'metrics': Dictionary of computed metrics
        - 'logits': Tensor of logits
        - 'labels': Tensor of labels
        - 'groups': Tensor of groups (or None)
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        for split_name, loader in holdout_loaders.items():
            all_logits = []
            all_labels = []
            all_groups = []
            
            for batch in loader:
                if is_bert:
                    # BERT input format
                    if len(batch) >= 3:
                        x, y, *rest = batch
                        groups = rest[0] if len(rest) > 0 else None
                    else:
                        x, y = batch
                        groups = None
                    
                    # Handle BERT input
                    if isinstance(x, torch.Tensor) and x.dim() == 3:
                        input_ids = x[:, :, 0].to(device)
                        attention_mask = x[:, :, 1].to(device)
                        token_type_ids = x[:, :, 2].to(device) if x.shape[2] > 2 else None
                        logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                        )
                        if hasattr(logits, 'logits'):
                            logits = logits.logits
                    else:
                        x = x.to(device)
                        logits = model(x)
                else:
                    # Image input format
                    if len(batch) >= 3:
                        x, y, *rest = batch
                        groups = rest[0] if len(rest) > 0 else None
                    else:
                        x, y = batch
                        groups = None
                    
                    x = x.to(device)
                    logits = model(x)
                
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
                if groups is not None:
                    all_groups.append(groups.cpu())
            
            logits = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0)
            groups = torch.cat(all_groups, dim=0) if all_groups else None
            
            # Compute metrics
            metrics = compute_all_metrics(logits, labels, groups=groups)
            
            # Return both metrics and raw data for logging
            results[split_name] = {
                'metrics': metrics,
                'logits': logits,
                'labels': labels,
                'groups': groups,
            }
    
    return results
