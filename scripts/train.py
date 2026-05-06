"""
Training script for MARLA (Misclassification Aware Rank-Limited Adaptation) and ERM.

This script trains MARLA models on pre-extracted embeddings, following the
two-stage training paradigm:
1. Stage 1: Train base classifier (not included here, assumes pre-trained)
2. Stage 2: Train MARLA adjustment on embeddings (this script)

Can work in two modes:
- Pre-extracted embeddings: Provide paths to embedding files
- On-the-fly extraction: Provide checkpoint path and dataset info (extracts embeddings internally)
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Add parent directory to path for marla package imports
script_dir = Path(__file__).parent.absolute()
package_dir = script_dir.parent.absolute()
parent_dir = package_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from marla.models import MARLAModel
from marla.models.architectures import get_model, get_classifier_and_feature_extractor
from marla.losses import (
    get_marla_weights,
    rebalance_weights,
    compute_spectral_decomposition,
    marla_loss,
    erm_loss,
    erm,
    marla,
)
from marla.optimizers import sgd_optimizer, cosine_lr_scheduler, constant_lr_scheduler
from marla.utils import MARLALogger, compute_all_metrics
from marla.utils.embedding_extraction import extract_embeddings_with_classifier
from marla.data import get_transform, SpuriousDataset, MultiNLIDataset, CivilCommentsWILDSDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MARLA model on embeddings')
    
    # Mode selection: Either provide embeddings OR checkpoint
    embedding_group = parser.add_argument_group('Pre-extracted embeddings mode')
    checkpoint_group = parser.add_argument_group('Checkpoint-based extraction mode')
    
    # Simplified single-file embedding argument (shortcut)
    embedding_group.add_argument('--embeddings', type=str, default=None,
                       help='Path to embeddings file (.pt file) - loads all data from single file (shortcut for --train_embeddings, etc.)')
    
    # Pre-extracted embeddings arguments (optional if checkpoint provided)
    embedding_group.add_argument('--train_embeddings', type=str, default=None,
                       help='Path to training embeddings (.pt file)')
    embedding_group.add_argument('--train_labels', type=str, default=None,
                       help='Path to training labels (.pt file)')
    embedding_group.add_argument('--train_groups', type=str, default=None,
                       help='Path to training groups (.pt file, optional)')
    embedding_group.add_argument('--val_embeddings', type=str, default=None,
                       help='Path to validation embeddings (.pt file)')
    embedding_group.add_argument('--val_labels', type=str, default=None,
                       help='Path to validation labels (.pt file)')
    embedding_group.add_argument('--val_groups', type=str, default=None,
                       help='Path to validation groups (.pt file, optional)')
    embedding_group.add_argument('--test_embeddings', type=str, default=None,
                       help='Path to test embeddings (.pt file)')
    embedding_group.add_argument('--test_labels', type=str, default=None,
                       help='Path to test labels (.pt file)')
    embedding_group.add_argument('--test_groups', type=str, default=None,
                       help='Path to test groups (.pt file, optional)')
    
    # Model arguments (optional if checkpoint provided)
    embedding_group.add_argument('--base_weight', type=str, default=None,
                       help='Path to base classifier weight matrix (.pt file)')
    embedding_group.add_argument('--base_bias', type=str, default=None,
                       help='Path to base classifier bias vector (.pt file)')
    
    # Checkpoint-based extraction arguments
    checkpoint_group.add_argument('--checkpoint', type=str, default=None,
                       help='Path to Stage 1 checkpoint (.pt file) - if provided, extracts embeddings on-the-fly')
    checkpoint_group.add_argument('--model', type=str, default=None,
                       choices=['imagenet_resnet50_pretrained', 'imagenet_resnet50', 
                               'bert-base-uncased', 'bert'],
                       help='Model architecture (required if --checkpoint provided)')
    checkpoint_group.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (required if --checkpoint provided)')
    checkpoint_group.add_argument('--dataset', type=str, default=None,
                       choices=['celeba', 'waterbirds', 'civilcomments', 'multinli'],
                       help='Dataset name (required if --checkpoint provided)')
    checkpoint_group.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset directory (required if --checkpoint provided)')
    checkpoint_group.add_argument('--data_transform', type=str, default='NoAugWaterbirdsCelebATransform',
                       help='Data transform for embedding extraction (default: NoAugWaterbirdsCelebATransform)')
    checkpoint_group.add_argument('--extract_batch_size', type=int, default=32,
                       help='Batch size for embedding extraction (default: 32)')
    checkpoint_group.add_argument('--extract_num_workers', type=int, default=0,
                       help='Number of data loader workers for extraction (default: 0)')
    
    # Training arguments
    parser.add_argument('--method', type=str, default='marla', choices=['marla', 'erm'],
                       help='Training method: marla (Misclassification Aware Rank-Limited Adaptation) or erm (Empirical Risk Minimization)')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='Number of training epochs (default: 1000 for Stage 2)')
    parser.add_argument('--batch_size', type=int, default=-1,
                       help='Batch size for training (-1 for full batch, like AFR)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.0,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--reg_coeff', type=float, default=0.0,
                       help='Regularization coefficient for adjustment matrix (MARLA only)')
    parser.add_argument('--grad_norm', type=float, default=1.0,
                       help='Gradient clipping norm (default: 1.0)')
    
    # MARLA-specific arguments
    parser.add_argument('--gamma', type=float, default=10.0,
                       help='Reweighting strength parameter (MARLA only)')
    parser.add_argument('--spectral_rank', type=int, default=10,
                       help='Spectral rank (number of eigenvectors, MARLA only)')
    parser.add_argument('--exp_weight_factor', type=float, default=1.0,
                       help='Exponential weighting factor for validation metric calculation (default: 1.0, used for selection only, not training)')
    
    # Logging arguments
    parser.add_argument('--output_dir', type=str, default='./logs/marla_experiment',
                       help='Output directory for logs and checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='marla',
                       help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='W&B run name')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eval_freq', type=int, default=1,
                       help='Evaluation frequency (every N epochs) - Stage 2 evaluates every epoch')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Checkpoint saving frequency (every N epochs) - Stage 2 does not save checkpoints')
    
    args = parser.parse_args()
    
    # Handle --embeddings shortcut: if provided, set all paths to the same file
    if args.embeddings is not None:
        if args.train_embeddings is not None:
            parser.error("Cannot specify both --embeddings and --train_embeddings. Use --embeddings alone for single-file mode.")
        # Set all paths to the same embeddings file
        emb_path = args.embeddings
        args.train_embeddings = emb_path
        args.train_labels = emb_path
        args.train_groups = emb_path
        args.val_embeddings = emb_path
        args.val_labels = emb_path
        args.val_groups = emb_path
        args.test_embeddings = emb_path
        args.test_labels = emb_path
        args.test_groups = emb_path
        args.base_weight = emb_path
        args.base_bias = emb_path
    
    # Validate arguments: must provide either embeddings OR checkpoint
    has_embeddings = args.train_embeddings is not None or args.embeddings is not None
    has_checkpoint = args.checkpoint is not None
    
    if not has_embeddings and not has_checkpoint:
        parser.error("Must provide either --checkpoint (for on-the-fly extraction) OR --embeddings/--train_embeddings (for pre-extracted embeddings)")
    
    if has_checkpoint:
        # Validate checkpoint mode arguments
        if args.model is None:
            parser.error("--model is required when --checkpoint is provided")
        if args.num_classes is None:
            parser.error("--num_classes is required when --checkpoint is provided")
        if args.dataset is None:
            parser.error("--dataset is required when --checkpoint is provided")
        if args.data_dir is None:
            parser.error("--data_dir is required when --checkpoint is provided")
    
    return args


def get_data_loaders_for_extraction(args):
    """
    Get data loaders for embedding extraction (reused from extract_embeddings.py).
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
    """
    # Get transform (usually no augmentation for embedding extraction)
    transform = get_transform(args.data_transform, train=False)
    
    # Create datasets based on dataset type
    if args.dataset in ['celeba', 'waterbirds']:
        # Use SpuriousDataset for image datasets
        train_dataset = SpuriousDataset(
            basedir=args.data_dir,
            split="train",
            transform=transform,
            prop=1.0,
            max_prop=1.0,
            seed=0  # Use fixed seed for consistency
        )
        
        val_dataset = SpuriousDataset(
            basedir=args.data_dir,
            split="val",
            transform=transform,
            prop=1.0,
            max_prop=1.0,
            seed=0
        )
        
        test_dataset = SpuriousDataset(
            basedir=args.data_dir,
            split="test",
            transform=transform,
            prop=1.0,
            max_prop=1.0,
            seed=0
        )
        
    elif args.dataset == 'civilcomments':
        # Use WILDS for CivilComments
        try:
            from wilds import get_dataset
        except ImportError:
            raise ImportError(
                "WILDS package not found. Install with: pip install wilds"
            )
        
        # Get parent directory of civilcomments (WILDS expects root_dir)
        civilcomments_dir = args.data_dir
        root_dir = os.path.dirname(civilcomments_dir)
        
        # Load WILDS dataset
        wilds_dataset = get_dataset(
            dataset='civilcomments',
            root_dir=root_dir,
            download=False
        )
        
        # Create transforms
        if args.data_transform == 'BertTokenizeTransform':
            from marla.data import BertTokenizeTransform
            transform_wilds = BertTokenizeTransform(train=False)
        else:
            transform_wilds = None
        
        train_dataset = CivilCommentsWILDSDataset(
            wilds_dataset, split="train", transform=transform_wilds
        )
        val_dataset = CivilCommentsWILDSDataset(
            wilds_dataset, split="val", transform=transform_wilds
        )
        test_dataset = CivilCommentsWILDSDataset(
            wilds_dataset, split="test", transform=transform_wilds
        )
        
    elif args.dataset == 'multinli':
        # Use MultiNLIDataset
        metadata_file = getattr(args, 'metadata_file', 'metadata_random.csv')
        
        train_dataset = MultiNLIDataset(
            root_dir=args.data_dir,
            split="train",
            transform=None,  # MultiNLI uses pre-tokenized features
            metadata_file=metadata_file
        )
        
        val_dataset = MultiNLIDataset(
            root_dir=args.data_dir,
            split="val",
            transform=None,
            metadata_file=metadata_file
        )
        
        test_dataset = MultiNLIDataset(
            root_dir=args.data_dir,
            split="test",
            transform=None,
            metadata_file=metadata_file
        )
        
    else:
        raise ValueError(
            f"Unknown dataset: {args.dataset}. "
            f"Supported: celeba, waterbirds, civilcomments, multinli"
        )
    
    # Create data loaders (no shuffling for embedding extraction)
    loader_kwargs = {
        'batch_size': args.extract_batch_size,
        'num_workers': args.extract_num_workers,
        'pin_memory': True,
        'shuffle': False
    }
    
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    
    return train_loader, val_loader, test_loader


def extract_embeddings_from_checkpoint(args, device):
    """
    Extract embeddings from Stage 1 checkpoint (similar to extract_embeddings.py).
    
    Returns:
        (train_emb, train_y, train_g), (val_emb, val_y, val_g), (test_emb, test_y, test_g)
        base_weight, base_bias
    """
    print(f"Loading model from {args.checkpoint}...")
    model = get_model(args.model, args.num_classes)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    
    model = model.to(device)
    model.eval()
    
    # Split into feature extractor and classifier
    print("Splitting model into feature extractor and classifier...")
    feature_extractor, classifier = get_classifier_and_feature_extractor(model)
    feature_extractor = feature_extractor.to(device)
    classifier = classifier.to(device)
    
    # Get base classifier weights and bias
    base_weight = classifier.weight.data.clone().cpu()
    base_bias = classifier.bias.data.clone().cpu()
    
    # Get data loaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_data_loaders_for_extraction(args)
    
    # Check if BERT model
    is_bert = 'bert' in args.model.lower() or args.dataset in ['civilcomments', 'multinli']
    
    # Extract embeddings
    print("Extracting embeddings from training set...")
    train_data = extract_embeddings_with_classifier(
        feature_extractor, classifier, train_loader, device, is_bert=is_bert
    )
    
    print("Extracting embeddings from validation set...")
    val_data = extract_embeddings_with_classifier(
        feature_extractor, classifier, val_loader, device, is_bert=is_bert
    )
    
    print("Extracting embeddings from test set...")
    test_data = extract_embeddings_with_classifier(
        feature_extractor, classifier, test_loader, device, is_bert=is_bert
    )
    
    # Unpack data tuples
    train_emb, train_pred, train_y, train_g = train_data
    val_emb, val_pred, val_y, val_g = val_data
    test_emb, test_pred, test_y, test_g = test_data
    
    # Store base weights/bias in args for initialize_model
    args.base_weight = base_weight
    args.base_bias = base_bias
    
    return (train_emb, train_y, train_g), (val_emb, val_y, val_g), (test_emb, test_y, test_g)


def load_data(args, device):
    """
    Load embeddings, labels, and groups.
    
    If checkpoint is provided, extracts embeddings on-the-fly.
    Otherwise, loads from pre-extracted embedding files.
    """
    # If checkpoint provided, extract embeddings on-the-fly
    if args.checkpoint is not None:
        return extract_embeddings_from_checkpoint(args, device)
    
    # Otherwise, load from pre-extracted files
    def load_tensor(path):
        if path is None:
            return None
        data = torch.load(path)
        # If path points to a dictionary (from extract_embeddings.py),
        # extract the appropriate key
        if isinstance(data, dict):
            # Try to infer which key to use based on path
            if 'train' in path.lower() or 'e' in data:
                return data.get('e', data.get('train_emb', None))
            elif 'val' in path.lower() or 'val_e' in data:
                return data.get('val_e', data.get('val_emb', None))
            elif 'test' in path.lower() or 'test_e' in data:
                return data.get('test_e', data.get('test_emb', None))
            elif 'label' in path.lower() or 'y' in data:
                return data.get('y', data.get('labels', None))
            elif 'group' in path.lower() or 'g' in data:
                return data.get('g', data.get('groups', None))
            elif 'weight' in path.lower() or 'w0' in data:
                return data.get('w0', data.get('weight', None))
            elif 'bias' in path.lower() or 'b0' in data:
                return data.get('b0', data.get('bias', None))
            else:
                # Return first tensor value
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        return value
                return None
        return data
    
    # Check if all paths point to the same file
    all_paths = [
        args.train_embeddings, args.train_labels, args.train_groups,
        args.val_embeddings, args.val_labels, args.val_groups,
        args.test_embeddings, args.test_labels, args.test_groups,
        args.base_weight, args.base_bias
    ]
    unique_paths = set([p for p in all_paths if p is not None])
    
    if len(unique_paths) == 1:
        # Load from single dictionary file
        emb_dict = torch.load(list(unique_paths)[0])
        train_emb = emb_dict.get('e', emb_dict.get('train_emb'))
        train_y = emb_dict.get('y', emb_dict.get('train_y'))
        train_g = emb_dict.get('g', emb_dict.get('train_g'))
        
        val_emb = emb_dict.get('val_e', emb_dict.get('val_emb'))
        val_y = emb_dict.get('val_y')
        val_g = emb_dict.get('val_g')
        
        test_emb = emb_dict.get('test_e', emb_dict.get('test_emb'))
        test_y = emb_dict.get('test_y')
        test_g = emb_dict.get('test_g')
        
        # Extract groups from metadata if 'g' not present but 'm' (metadata) is present
        # Handle AFR format embeddings for CivilComments (legacy format)
        if train_g is None and 'm' in emb_dict:
            train_m = emb_dict['m']
            # For CivilComments: groups = y * 2 + black (where black is metadata[:, 6])
            # Or more generally: groups = y * 2 + argmax(identity_metadata)
            if train_m.dim() == 2 and train_m.shape[1] >= 8:
                # Use black-white grouping: y * 2 + black
                black = train_m[:, 6]  # Column 6 is 'black'
                train_g = train_y * 2 + black
            else:
                train_g = None
        
        if val_g is None and 'val_m' in emb_dict:
            val_m = emb_dict['val_m']
            if val_m.dim() == 2 and val_m.shape[1] >= 8:
                black = val_m[:, 6]
                val_g = val_y * 2 + black
            else:
                val_g = None
        
        if test_g is None and 'test_m' in emb_dict:
            test_m = emb_dict['test_m']
            if test_m.dim() == 2 and test_m.shape[1] >= 8:
                black = test_m[:, 6]
                test_g = test_y * 2 + black
            else:
                test_g = None
        
        # Override base_weight and base_bias from dict
        if 'w0' in emb_dict:
            args.base_weight = None  # Will be loaded from dict
        if 'b0' in emb_dict:
            args.base_bias = None  # Will be loaded from dict
    else:
        # Load from separate files
        train_emb = load_tensor(args.train_embeddings)
        train_y = load_tensor(args.train_labels)
        train_g = load_tensor(args.train_groups)
        
        val_emb = load_tensor(args.val_embeddings)
        val_y = load_tensor(args.val_labels)
        val_g = load_tensor(args.val_groups)
        
        test_emb = load_tensor(args.test_embeddings)
        test_y = load_tensor(args.test_labels)
        test_g = load_tensor(args.test_groups)
    
    return (train_emb, train_y, train_g), (val_emb, val_y, val_g), (test_emb, test_y, test_g)


def initialize_model(args, train_embeddings, train_labels, device):
    """Initialize MARLA or ERM model."""
    # Load base classifier weights
    # Check if base_weight/base_bias are paths, tensors, or already loaded from dict
    if isinstance(args.base_weight, str):
        base_weight = torch.load(args.base_weight).to(device)
    elif args.base_weight is None:
        # Try to load from train_embeddings file if it's a dict
        # Handle case where extract_embeddings.py saved everything in one file
        if hasattr(args, '_emb_dict') and 'w0' in args._emb_dict:
            base_weight = args._emb_dict['w0'].to(device)
        else:
            raise ValueError("base_weight not provided and not found in embedding dict")
    elif isinstance(args.base_weight, torch.Tensor):
        # Already a tensor (from checkpoint extraction)
        base_weight = args.base_weight.to(device)
    else:
        base_weight = args.base_weight.to(device)
    
    if isinstance(args.base_bias, str):
        base_bias = torch.load(args.base_bias).to(device)
    elif args.base_bias is None:
        if hasattr(args, '_emb_dict') and 'b0' in args._emb_dict:
            base_bias = args._emb_dict['b0'].to(device)
        else:
            raise ValueError("base_bias not provided and not found in embedding dict")
    elif isinstance(args.base_bias, torch.Tensor):
        # Already a tensor (from checkpoint extraction)
        base_bias = args.base_bias.to(device)
    else:
        base_bias = args.base_bias.to(device)
    
    if args.method == 'marla':
        # Compute initial logits with base classifier
        with torch.no_grad():
            init_logits = F.linear(train_embeddings, base_weight, base_bias)
        
        # Compute MARLA weights (with rebalance=True)
        weights = get_marla_weights(init_logits, train_labels, gamma=args.gamma, rebalance=True)
        
        # Compute spectral decomposition
        print(f"Computing spectral decomposition with rank={args.spectral_rank}...")
        V_k, eigenvalues = compute_spectral_decomposition(
            train_embeddings, weights, k=args.spectral_rank
        )
        print(f"Top-{args.spectral_rank} eigenvalues: {eigenvalues[:5].tolist()}...")
        
        # Create MARLA model
        model = MARLAModel(base_weight, base_bias, V_k.to(device)).to(device)
        
        return model, weights, eigenvalues
    else:
        # ERM: Use simple linear model initialized with base weights
        # More stable and prevents overfitting
        num_classes, embedding_dim = base_weight.shape
        model = torch.nn.Linear(embedding_dim, num_classes).to(device)
        model.weight.data = base_weight.clone()
        model.bias.data = base_bias.clone()
        
        return model, None, None


def train_epoch(model, train_loader, criterion, optimizer, scheduler, args, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_logits = []
    
    for batch in train_loader:
        # Match AFR: zero_grad before forward pass
        optimizer.zero_grad()
        
        if args.method == 'marla':
            embeddings, labels, weights = batch
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            
            # Forward pass
            logits = model(embeddings)
            all_logits.append(logits)
            
            # Compute loss with regularization
            adjustment_matrix = model.adjustment
            loss = criterion(
                logits, labels, weights,
                reg_coeff=args.reg_coeff,
                adjustment_matrix=adjustment_matrix
            )
        else:
            # ERM
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(embeddings)
            all_logits.append(logits)
            
            # Compute loss
            loss = criterion(logits, labels)
        
        loss.backward()
        # Gradient clipping if grad_norm > 0
        if hasattr(args, 'grad_norm') and args.grad_norm > 0:
            from torch.nn.utils import clip_grad_norm_
            clip_grad_norm_(model.parameters(), max_norm=args.grad_norm)
        optimizer.step()
        # Step scheduler after each batch/epoch
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    # Concatenate all logits if multiple batches
    if len(all_logits) > 1:
        all_logits = torch.cat(all_logits, dim=0)
    else:
        all_logits = all_logits[0] if all_logits else None
    
    return total_loss / n_batches if n_batches > 0 else 0.0, all_logits


def evaluate(model, embeddings, labels, groups, device):
    """Evaluate model on a dataset."""
    model.eval()
    with torch.no_grad():
        # Embeddings and labels should already be on device, but ensure they are
        if embeddings.device != device:
            embeddings = embeddings.to(device)
        if labels.device != device:
            labels = labels.to(device)
        
        logits = model(embeddings)
        
        if groups is not None and groups.device != device:
            groups = groups.to(device)
        
        return logits, labels, groups


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data (or extract from checkpoint)
    if args.checkpoint is not None:
        print("Extracting embeddings from checkpoint...")
    else:
        print("Loading pre-extracted embeddings...")
    (train_emb, train_y, train_g), (val_emb, val_y, val_g), (test_emb, test_y, test_g) = load_data(args, device)
    
    # Store embedding dict if loaded from single file (for base_weight/bias access)
    # Only needed if not using checkpoint mode (checkpoint mode already sets base_weight/bias)
    if args.checkpoint is None:
        all_paths = [
            args.train_embeddings, args.train_labels, args.train_groups,
            args.val_embeddings, args.val_labels, args.val_groups,
            args.test_embeddings, args.test_labels, args.test_groups,
            args.base_weight, args.base_bias
        ]
        unique_paths = set([p for p in all_paths if p is not None])
        if len(unique_paths) == 1:
            args._emb_dict = torch.load(list(unique_paths)[0])
            # Update base_weight and base_bias to point to dict
            if 'w0' in args._emb_dict:
                args.base_weight = None  # Will be loaded from dict in initialize_model
            if 'b0' in args._emb_dict:
                args.base_bias = None  # Will be loaded from dict in initialize_model
    
    # For MARLA: Split training data into 80% reweighting and 20% training
    # Use reweighting set for
    # computing weights/spectral decomposition, and training set is used for training
    if args.method == 'marla':
        n_total = train_emb.shape[0]
        n_reweighting = int(0.8 * n_total)
        
        # Create random permutation for splitting
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
        reweighting_idx = indices[:n_reweighting]
        train_idx = indices[n_reweighting:]
        
        # Split embeddings, labels, and groups
        reweighting_emb = train_emb[reweighting_idx]
        reweighting_y = train_y[reweighting_idx]
        reweighting_g = train_g[reweighting_idx] if train_g is not None else None
        
        train_emb_split = train_emb[train_idx]
        train_y_split = train_y[train_idx]
        train_g_split = train_g[train_idx] if train_g is not None else None
        
        print(f"Split training data: {n_reweighting:,} (80%) for reweighting, {len(train_idx):,} (20%) for training")
        
        # Move to device
        reweighting_emb = reweighting_emb.to(device)
        reweighting_y = reweighting_y.to(device)
        train_emb_split = train_emb_split.to(device)
        train_y_split = train_y_split.to(device)
        val_emb = val_emb.to(device)
        val_y = val_y.to(device)
        test_emb = test_emb.to(device)
        test_y = test_y.to(device)
        
        if reweighting_g is not None:
            reweighting_g = reweighting_g.to(device)
        if train_g_split is not None:
            train_g_split = train_g_split.to(device)
        if val_g is not None:
            val_g = val_g.to(device)
        if test_g is not None:
            test_g = test_g.to(device)
        
        # Initialize model using reweighting set for weights/spectral decomposition
        print(f"Initializing {args.method.upper()} model...")
        model, reweighting_weights, eigenvalues = initialize_model(args, reweighting_emb, reweighting_y, device)
        
        # Compute weights for training set (needed for training loop)
        # Use the same base weights/bias that were used for reweighting
        with torch.no_grad():
            train_init_logits = F.linear(train_emb_split, model.base_weight, model.base_bias)
        train_weights = get_marla_weights(train_init_logits, train_y_split, gamma=args.gamma, rebalance=True)
        train_weights = train_weights.to(device)
        
        # Use training split for actual training
        train_emb = train_emb_split
        train_y = train_y_split
        train_g = train_g_split
    else:
        # ERM: Use full training set (no split needed)
        train_emb = train_emb.to(device)
        train_y = train_y.to(device)
        val_emb = val_emb.to(device)
        val_y = val_y.to(device)
        test_emb = test_emb.to(device)
        test_y = test_y.to(device)
        
        if train_g is not None:
            train_g = train_g.to(device)
        if val_g is not None:
            val_g = val_g.to(device)
        if test_g is not None:
            test_g = test_g.to(device)
        
        # Initialize model
        print(f"Initializing {args.method.upper()} model...")
        model, train_weights, eigenvalues = initialize_model(args, train_emb, train_y, device)
    
    # Create data loaders
    # Use full batch training if batch_size is -1 (like AFR)
    if args.batch_size == -1:
        # Full batch: return single batch with all data
        if args.method == 'marla':
            train_loader = [(train_emb, train_y, train_weights)]
        else:
            train_loader = [(train_emb, train_y)]
    else:
        # Batched training
        if args.method == 'marla':
            # MARLA: Include weights in dataset
            train_dataset = TensorDataset(train_emb, train_y, train_weights)
        else:
            # ERM: No weights
            train_dataset = TensorDataset(train_emb, train_y)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Setup optimizer and scheduler
    # Use constant_lr_scheduler for MARLA
    optimizer = sgd_optimizer(model, args.lr, args.momentum, args.weight_decay)
    scheduler = constant_lr_scheduler()  # Match AFR: always use constant LR for spectral_afr
    
    # Setup loss function
    if args.method == 'marla':
        criterion = marla()
    else:
        criterion = erm()
    
    # Setup logger
    config = {
        'method': args.method,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'reg_coeff': args.reg_coeff,
        'gamma': args.gamma if args.method == 'marla' else None,
        'spectral_rank': args.spectral_rank if args.method == 'marla' else None,
        'seed': args.seed,
        'exp_weight_factor': getattr(args, 'exp_weight_factor', None),
    }
    
    logger = MARLALogger(
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"{args.method}_{args.seed}",
        config=config
    )
    
    # Log initial eigenvalues if MARLA
    if args.method == 'marla' and eigenvalues is not None:
        variance_captured = eigenvalues[:args.spectral_rank].sum().item()
        logger.logger.info(f"Eigenvalue variance captured: {variance_captured:.4f}")
        if args.use_wandb:
            import wandb
            wandb.log({"eigenvalue_variance_captured": variance_captured}, step=0)
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    start_time = time.time()
    best_val_wga = 0.0
    best_test_at_val = 0.0
    best_val_wga_epoch = None  # Epoch with best validation WGA
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss, train_logits_from_epoch = train_epoch(model, train_loader, criterion, optimizer, scheduler, args, device)
        
        # Evaluate at every epoch (Stage 2: evaluate every epoch, no checkpoints)
        # Use logits from training epoch if available (more efficient), otherwise recompute
        if train_logits_from_epoch is not None:
            train_logits = train_logits_from_epoch
            train_labels_eval = train_y
            train_groups_eval = train_g
        else:
            train_logits, train_labels_eval, train_groups_eval = evaluate(
                model, train_emb, train_y, train_g, device
            )
        logger.log_metrics(
            epoch, train_logits, train_labels_eval, 'train',
            groups=train_groups_eval
        )
        
        # Validation metrics
        val_logits, val_labels_eval, val_groups_eval = evaluate(
            model, val_emb, val_y, val_g, device
        )
        logger.log_metrics(
            epoch, val_logits, val_labels_eval, 'val',
            groups=val_groups_eval,
            exp_weight_factor=getattr(args, 'exp_weight_factor', None)
        )
        
        # Test metrics
        test_logits, test_labels_eval, test_groups_eval = evaluate(
            model, test_emb, test_y, test_g, device
        )
        logger.log_metrics(
            epoch, test_logits, test_labels_eval, 'test',
            groups=test_groups_eval
        )
        
        # Track best validation metrics for logging
        current_val_wga = logger.best_metrics['val']['worst_group_accuracy']
        current_test_wga = logger.best_metrics['test']['worst_group_accuracy']
        
        # Log "New best validation wga" in scientific notation when improved
        if current_val_wga > best_val_wga:
            logger.logger.info(f"\nNew best validation wga: {current_val_wga:1.3e}")
            best_val_wga = current_val_wga
            best_val_wga_epoch = epoch  # Track epoch with best validation WGA
            best_test_at_val = current_test_wga
            logger.logger.info(f"\nNew best test @ validation: {best_test_at_val:1.3e}")
            if epoch not in logger.metrics:
                logger.metrics[epoch] = {}
            logger.metrics[epoch]['best_test_at_val'] = best_test_at_val
        
        logger.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        logger.logger.info(f"  Best Val WGA: {best_val_wga:.4f}")
        
        # Log to W&B
        if args.use_wandb:
            import wandb
            wandb.log({"train_loss": train_loss, "epoch": epoch}, step=epoch)
    
    # Finalize
    elapsed_time = time.time() - start_time
    logger.logger.info(f"\nTraining completed in {elapsed_time/60:.2f} minutes")
    
    # Get final best test metrics
    best_test_wga = logger.best_metrics['test']['worst_group_accuracy']
    best_test_mean = logger.best_metrics['test']['mean_accuracy']
    best_test_balanced = logger.best_metrics['test']['class_balanced_accuracy']
    best_val_mean = logger.best_metrics['val']['mean_accuracy']
    
    # Final summary
    logger.logger.info(f"Best test at val:     {best_test_at_val:.3f}")
    logger.logger.info(f"Best test WGA:        {best_test_wga:.3f}")
    logger.logger.info(f"Best val WGA:         {best_val_wga:.3f}")
    if best_val_wga_epoch is not None:
        logger.logger.info(f"  (Best val WGA at epoch {best_val_wga_epoch})")
    logger.logger.info(f"Best test mean val:   {best_test_mean:.3f}")
    logger.logger.info(f"Best test mean:       {best_test_mean:.3f}")
    logger.logger.info(f"Best test balanced:   {best_test_balanced:.3f}")
    logger.logger.info(f"Best val mean:        {best_val_mean:.3f}")
    
    logger.finalize()
    
    print(f"\nTraining complete! Logs saved to: {args.output_dir}")
    print(f"Best Val WGA: {best_val_wga:.4f}")


if __name__ == '__main__':
    main()
