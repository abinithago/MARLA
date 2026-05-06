"""
Extract embeddings from Stage 1 trained model.

This script loads a trained Stage 1 model, splits it into feature extractor
and classifier, then extracts embeddings from train/val/test splits.
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add parent directory to path for marla package imports
script_dir = Path(__file__).parent.absolute()
package_dir = script_dir.parent.absolute()
parent_dir = package_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from marla.models.architectures import get_model, get_classifier_and_feature_extractor
from marla.utils.embedding_extraction import (
    extract_embeddings_with_classifier,
    save_embeddings_dict
)
from marla.data import get_transform, SpuriousDataset, MultiNLIDataset, CivilCommentsWILDSDataset
from torch.utils.data import DataLoader
import os


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract embeddings from Stage 1 model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Stage 1 checkpoint (.pt file)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['imagenet_resnet50_pretrained', 'imagenet_resnet50', 
                               'bert-base-uncased', 'bert'],
                       help='Model architecture (must match checkpoint)')
    parser.add_argument('--num_classes', type=int, required=True,
                       help='Number of classes')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['celeba', 'waterbirds', 'civilcomments', 'multinli'],
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--data_transform', type=str, default='NoAugWaterbirdsCelebATransform',
                       help='Data transform for embedding extraction')
    
    # Output arguments
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save embeddings (.pt file)')
    
    parser.add_argument('--train_prop', type=float, default=None,
                       help='Train proportion (negative means last X%%, e.g., -0.2 for last 20%%)')
    parser.add_argument('--val_prop', type=float, default=1.0,
                       help='Validation proportion (default: 1.0)')
    parser.add_argument('--max_prop', type=float, default=1.0,
                       help='Max proportion (default: 1.0)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for data splitting (default: 21 for CelebA Stage 2)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    return parser.parse_args()


def get_data_loaders(args):
    """
    Get data loaders for embedding extraction.
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
    """
    # Get transform (usually no augmentation for embedding extraction)
    transform = get_transform(args.data_transform, train=False)
    
    # Create datasets based on dataset type
    if args.dataset in ['celeba', 'waterbirds']:
        if args.train_prop is None:
            train_prop = -0.2 if args.dataset == 'celeba' else -20.0
        else:
            train_prop = args.train_prop
        
        if args.seed is None:
            seed = 21 if args.dataset == 'celeba' else 0
        else:
            seed = args.seed
        
        train_dataset = SpuriousDataset(
            basedir=args.data_dir,
            split="train",
            transform=transform,
            prop=train_prop,
            max_prop=args.max_prop,
            seed=seed
        )
        
        val_dataset = SpuriousDataset(
            basedir=args.data_dir,
            split="val",
            transform=transform,
            prop=args.val_prop,
            max_prop=args.max_prop,
            seed=seed  # Use same seed for consistency
        )
        
        test_dataset = SpuriousDataset(
            basedir=args.data_dir,
            split="test",
            transform=transform,
            prop=1.0,  # Test always uses full test set
            max_prop=args.max_prop,
            seed=seed  # Use same seed for consistency
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
        'batch_size': args.batch_size,
        'num_workers': getattr(args, 'num_workers', 4),
        'pin_memory': True,
        'shuffle': False
    }
    
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    
    return train_loader, val_loader, test_loader


def main(): 
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
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
    train_loader, val_loader, test_loader = get_data_loaders(args)
    
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
    
    # Save embeddings
    print(f"Saving embeddings to {args.output_path}...")
    save_embeddings_dict(
        train_data, val_data, test_data,
        base_weight, base_bias,
        args.output_path
    )
    
    print("Embedding extraction complete!")
    print(f"Next step: Train Stage 2 using train.py with --train_embeddings {args.output_path}")


if __name__ == '__main__':
    main()
