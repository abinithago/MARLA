"""
Embedding extraction utilities for MARLA.

Extracts embeddings from trained models for Stage 2 training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Optional


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_bert: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract embeddings, predictions, labels, and groups from a dataloader.
    
    Args:
        model: Model to extract embeddings from (should output embeddings, not logits)
        dataloader: DataLoader to extract from
        device: Device to run on
        is_bert: Whether this is a BERT model (handles input format differently)
    
    Returns:
        embeddings: Tensor of shape (n, d)
        predictions: Tensor of shape (n,)
        labels: Tensor of shape (n,)
        groups: Tensor of shape (n,) or None
    """
    model.eval()
    all_embeddings = []
    all_predictions = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            if is_bert:
                # BERT input format: (input_ids, attention_mask, token_type_ids) or stacked tensor
                if len(batch) >= 3:
                    x, y, *rest = batch
                    groups = rest[0] if len(rest) > 0 else None
                else:
                    x, y = batch
                    groups = None
                
                # Handle BERT input format
                if isinstance(x, torch.Tensor) and x.dim() == 3:
                    # Stacked format: (batch, seq_len, 3) -> separate tensors
                    input_ids = x[:, :, 0].to(device)
                    attention_mask = x[:, :, 1].to(device)
                    token_type_ids = x[:, :, 2].to(device) if x.shape[2] > 2 else None
                else:
                    # Assume separate tensors or dict
                    if isinstance(x, dict):
                        input_ids = x['input_ids'].to(device)
                        attention_mask = x['attention_mask'].to(device)
                        token_type_ids = x.get('token_type_ids', None)
                        if token_type_ids is not None:
                            token_type_ids = token_type_ids.to(device)
                    else:
                        # Fallback: assume first element is input_ids
                        input_ids = x.to(device)
                        attention_mask = torch.ones_like(input_ids)
                        token_type_ids = None
                
                embeddings = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            else:
                # Image input format
                if len(batch) >= 3:
                    x, y, *rest = batch
                    groups = rest[0] if len(rest) > 0 else None
                else:
                    x, y = batch
                    groups = None
                
                x = x.to(device)
                embeddings = model(x)
            
            # Get predictions (if model outputs logits, take argmax; if embeddings, skip)
            if embeddings.shape[-1] > 1000:  # Heuristic: logits have num_classes dim
                predictions = torch.argmax(embeddings, dim=1)
            else:
                # Model outputs embeddings, need classifier for predictions
                # Should be handled by caller providing classifier separately
                predictions = torch.zeros(embeddings.shape[0], dtype=torch.long)
            
            all_embeddings.append(embeddings.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(y.cpu())
            if groups is not None:
                all_groups.append(groups.cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    groups = torch.cat(all_groups, dim=0) if all_groups else None
    
    return embeddings, predictions, labels, groups


def extract_embeddings_with_classifier(
    feature_extractor: nn.Module,
    classifier: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_bert: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract embeddings using separate feature extractor and classifier.
    
    Args:
        feature_extractor: Model that outputs embeddings
        classifier: Linear layer that outputs logits from embeddings
        dataloader: DataLoader to extract from
        device: Device to run on
        is_bert: Whether this is a BERT model
    
    Returns:
        embeddings: Tensor of shape (n, d)
        predictions: Tensor of shape (n,)
        labels: Tensor of shape (n,)
        groups: Tensor of shape (n,) or None
    """
    feature_extractor.eval()
    classifier.eval()
    
    all_embeddings = []
    all_predictions = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            if is_bert:
                # BERT input format
                if len(batch) >= 3:
                    x, y, *rest = batch
                    groups = rest[0] if len(rest) > 0 else None
                else:
                    x, y = batch
                    groups = None
                
                # Handle BERT input format
                if isinstance(x, torch.Tensor) and x.dim() == 3:
                    input_ids = x[:, :, 0].to(device)
                    attention_mask = x[:, :, 1].to(device)
                    token_type_ids = x[:, :, 2].to(device) if x.shape[2] > 2 else None
                else:
                    if isinstance(x, dict):
                        input_ids = x['input_ids'].to(device)
                        attention_mask = x['attention_mask'].to(device)
                        token_type_ids = x.get('token_type_ids', None)
                        if token_type_ids is not None:
                            token_type_ids = token_type_ids.to(device)
                    else:
                        input_ids = x.to(device)
                        attention_mask = torch.ones_like(input_ids)
                        token_type_ids = None
                
                embeddings = feature_extractor(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            else:
                # Image input format
                if len(batch) >= 3:
                    x, y, *rest = batch
                    groups = rest[0] if len(rest) > 0 else None
                else:
                    x, y = batch
                    groups = None
                
                x = x.to(device)
                embeddings = feature_extractor(x)
            
            # Get logits and predictions
            logits = classifier(embeddings)
            predictions = torch.argmax(logits, dim=1)
            
            all_embeddings.append(embeddings.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(y.cpu())
            if groups is not None:
                all_groups.append(groups.cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    groups = torch.cat(all_groups, dim=0) if all_groups else None
    
    return embeddings, predictions, labels, groups


def save_embeddings_dict(
    train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    val_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    test_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    base_weight: torch.Tensor,
    base_bias: torch.Tensor,
    output_path: str
):
    """
    Save embeddings dictionary in the format expected by train.py.
    
    Args:
        train_data: (embeddings, predictions, labels, groups)
        val_data: (embeddings, predictions, labels, groups)
        test_data: (embeddings, predictions, labels, groups)
        base_weight: Base classifier weight matrix
        base_bias: Base classifier bias vector
        output_path: Path to save .pt file
    """
    train_emb, train_pred, train_y, train_g = train_data
    val_emb, val_pred, val_y, val_g = val_data
    test_emb, test_pred, test_y, test_g = test_data
    
    emb_dict = {
        'e': train_emb,  # Training embeddings
        'y': train_y,    # Training labels
        'pred': train_pred,  # Training predictions
        'g': train_g,    # Training groups (optional)
        'val_e': val_emb,
        'val_y': val_y,
        'val_pred': val_pred,
        'val_g': val_g,
        'test_e': test_emb,
        'test_y': test_y,
        'test_pred': test_pred,
        'test_g': test_g,
        'w0': base_weight,  # Base classifier weights
        'b0': base_bias,    # Base classifier bias
    }
    
    torch.save(emb_dict, output_path)
    print(f"Saved embeddings to {output_path}")
