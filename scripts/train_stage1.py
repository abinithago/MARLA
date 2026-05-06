"""
Stage 1 Training Script: Train ERM base classifier.

This script trains an ERM (Empirical Risk Minimization) base classifier
on the full dataset. The trained model is then used in Stage 2 for
embedding extraction and MARLA/ERM training.
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import random
from pathlib import Path

# Add parent directory to path for marla package imports
script_dir = Path(__file__).parent.absolute()
package_dir = script_dir.parent.absolute()
parent_dir = package_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from marla.models.architectures import get_model
from marla.losses import erm
from marla.optimizers import sgd_optimizer, cosine_lr_scheduler, constant_lr_scheduler
from marla.utils import MARLALogger
from marla.utils.training import train_epoch, eval_model
from marla.data import get_transform, SpuriousDataset, MultiNLIDataset, CivilCommentsWILDSDataset
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stage 1: Train ERM base classifier')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['celeba', 'waterbirds', 'civilcomments', 'multinli'],
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--data_transform', type=str, default='AugWaterbirdsCelebATransform',
                       help='Data transform name')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='imagenet_resnet50_pretrained',
                       choices=['imagenet_resnet50_pretrained', 'imagenet_resnet50', 
                               'bert-base-uncased', 'bert'],
                       help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (auto-detected if not provided)')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.003,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.0,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='constant_lr_scheduler',
                       choices=['constant_lr_scheduler', 'cosine_lr_scheduler'],
                       help='Learning rate scheduler')
    
    # Data arguments
    parser.add_argument('--train_prop', type=float, default=1.0,
                       help='Proportion of training data to use')
    parser.add_argument('--val_prop', type=float, default=1.0,
                       help='Proportion of validation data to use')
    parser.add_argument('--max_prop', type=float, default=1.0,
                       help='Maximum proportion of data to consider (for subsampling)')
    
    # Logging arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for logs and checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='marla_stage1',
                       help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='W&B run name')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eval_freq', type=int, default=10,
                       help='Evaluation frequency (every N epochs)')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Checkpoint saving frequency (every N epochs)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    return parser.parse_args()


def get_data_loaders(args):
    """
    Get data loaders for the specified dataset.
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
    """
    # Get transforms
    train_transform = get_transform(args.data_transform, train=True)
    test_transform = get_transform(args.data_transform, train=False)
    
    # Create datasets based on dataset type
    if args.dataset in ['celeba', 'waterbirds']:
        # Use SpuriousDataset for image datasets
        train_dataset = SpuriousDataset(
            basedir=args.data_dir,
            split="train",
            transform=train_transform,
            prop=args.train_prop,
            max_prop=getattr(args, 'max_prop', 1.0),
            seed=args.seed
        )
        
        val_dataset = SpuriousDataset(
            basedir=args.data_dir,
            split="val",
            transform=test_transform,
            prop=args.val_prop,
            max_prop=1.0,
            seed=args.seed
        )
        
        test_dataset = SpuriousDataset(
            basedir=args.data_dir,
            split="test",
            transform=test_transform,
            prop=1.0,
            max_prop=1.0,
            seed=args.seed
        )
        
        is_bert = False
        
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
        
        # Create train/val/test transforms
        # For WILDS, we need to use their transform system
        # For now, use BertTokenizeTransform if available
        if args.data_transform == 'BertTokenizeTransform':
            from marla.data import BertTokenizeTransform
            train_transform_wilds = BertTokenizeTransform(train=True)
            test_transform_wilds = BertTokenizeTransform(train=False)
        else:
            # Use WILDS default transform
            train_transform_wilds = None
            test_transform_wilds = None
        
        train_dataset = CivilCommentsWILDSDataset(
            wilds_dataset, split="train", transform=train_transform_wilds
        )
        val_dataset = CivilCommentsWILDSDataset(
            wilds_dataset, split="val", transform=test_transform_wilds
        )
        test_dataset = CivilCommentsWILDSDataset(
            wilds_dataset, split="test", transform=test_transform_wilds
        )
        
        is_bert = True
        
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
        
        is_bert = True
        
    else:
        raise ValueError(
            f"Unknown dataset: {args.dataset}. "
            f"Supported: celeba, waterbirds, civilcomments, multinli"
        )
    
    # Create data loaders
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': getattr(args, 'num_workers', 4),
        'pin_memory': True,
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader


def main():
    args = parse_args()
    
    # Set random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)  # Python's random module (used by SpuriousDataset)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(args)
    
    # Determine if BERT model (needed for evaluation)
    is_bert = 'bert' in args.model.lower() or args.dataset in ['civilcomments', 'multinli']
    
    # Determine number of classes
    if args.num_classes is None:
        if hasattr(train_loader.dataset, 'n_classes'):
            num_classes = train_loader.dataset.n_classes
        elif hasattr(train_loader.dataset, 'num_classes'):
            num_classes = train_loader.dataset.num_classes
        else:
            raise ValueError(f"Dataset {type(train_loader.dataset)} does not have n_classes or num_classes attribute")
    else:
        num_classes = args.num_classes
    
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    print(f"Initializing {args.model} model...")
    model = get_model(args.model, num_classes).to(device)
    
    # Setup optimizer and scheduler
    optimizer = sgd_optimizer(model, args.lr, args.momentum, args.weight_decay)
    
    if args.scheduler == 'cosine_lr_scheduler':
        scheduler = cosine_lr_scheduler(optimizer, args.num_epochs)
    else:
        scheduler = constant_lr_scheduler()
    
    # Setup loss function
    criterion = erm()
    
    # Setup logger
    config = {
        'dataset': args.dataset,
        'model': args.model,
        'num_classes': num_classes,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'seed': args.seed,
    }
    
    logger = MARLALogger(
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"stage1_{args.dataset}_{args.seed}",
        config=config
    )
    
    # Training loop
    print(f"\nStarting Stage 1 training for {args.num_epochs} epochs...")
    start_time = time.time()
    best_val_wga = 0.0
    
    for epoch in range(args.num_epochs):
        # Train
        loss_meter, acc_groups = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Log train results from training loop
        n_spurious = getattr(train_loader.dataset, 'n_spurious', 1)
        logger.log_train_results_from_acc_groups(epoch, acc_groups, split='train', n_spurious=n_spurious)
        
        # Evaluate
        if epoch % args.eval_freq == 0 or epoch == args.num_epochs - 1:
            
            # Evaluate on val and test sets
            holdout_loaders = {'val': val_loader, 'test': test_loader}
            results_dict = eval_model(model, holdout_loaders, device, is_bert=is_bert)
            
            # Log metrics using MARLALogger (logs all 4 metrics: mean acc, balanced acc, WGA, worst percentile)
            for split_name, result_data in results_dict.items():
                logits = result_data['logits']
                labels = result_data['labels']
                groups = result_data['groups']
                
                # Use logger.log_metrics() which automatically computes and logs all metrics
                logger.log_metrics(
                    epoch, logits, labels, split_name,
                    groups=groups
                )
            
            # Determine if this is the best model based on val WGA
            current_val_wga = logger.best_metrics['val']['worst_group_accuracy']
            is_best = (epoch == 0) or (current_val_wga > best_val_wga)
            if is_best:
                best_val_wga = current_val_wga
            
            # Save checkpoint
            if epoch % args.save_freq == 0 or epoch == args.num_epochs - 1 or is_best:
                logger.save_checkpoint(model, epoch, is_best=is_best)
        
        logger.logger.info(f"Epoch {epoch}: Train Loss = {loss_meter.avg:.4f}")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Save final checkpoint
    logger.save_checkpoint(model, args.num_epochs - 1, is_best=False)
    
    # Finalize
    elapsed_time = time.time() - start_time
    logger.logger.info(f"\nStage 1 training completed in {elapsed_time/60:.2f} minutes")
    logger.finalize()
    
    print(f"\nStage 1 training complete! Model saved to: {args.output_dir}")
    print(f"Next step: Extract embeddings using extract_embeddings.py")


if __name__ == '__main__':
    main()
