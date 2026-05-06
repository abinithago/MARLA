#!/usr/bin/env python3
"""
Hyperparameter sweep script for MARLA.

Sweeps over gamma and rank values for a given dataset and collects results.
Supports both checkpoint-based and embedding-based training modes.
"""

import argparse
import os
import sys
import subprocess
import json
import time
from pathlib import Path
from itertools import product
from typing import List, Dict, Any
import torch

# Add parent directory to path for marla package imports
script_dir = Path(__file__).parent.absolute()
package_dir = script_dir.parent.absolute()
parent_dir = package_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter sweep for MARLA over gamma and rank values'
    )
    
    # Dataset and data arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['celeba', 'waterbirds', 'civilcomments', 'multinli'],
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    
    # Mode selection: checkpoint-based or embedding-based
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--checkpoint', type=str, default=None,
                           help='Path to Stage 1 checkpoint (.pt file) - checkpoint-based mode')
    mode_group.add_argument('--embedding_dir', type=str, default=None,
                           help='Directory containing pre-extracted embeddings - embedding-based mode')
    
    # Model arguments (required for checkpoint mode)
    parser.add_argument('--model', type=str, default=None,
                       choices=['imagenet_resnet50_pretrained', 'imagenet_resnet50', 
                               'bert-base-uncased', 'bert'],
                       help='Model architecture (required for checkpoint mode)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (required for checkpoint mode)')
    parser.add_argument('--data_transform', type=str, default='NoAugWaterbirdsCelebATransform',
                       help='Data transform for embedding extraction')
    
    # Hyperparameter sweep arguments
    parser.add_argument('--gamma_list', type=str, required=True,
                       help='Comma-separated list of gamma values (e.g., "5.0,10.0,15.0")')
    parser.add_argument('--rank_list', type=str, required=True,
                       help='Comma-separated list of rank values (e.g., "1,5,10,20")')
    
    # Training arguments (passed to train.py)
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=-1,
                       help='Batch size for training (-1 for full batch)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.0,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--reg_coeff', type=float, default=0.0,
                       help='Regularization coefficient for adjustment matrix')
    parser.add_argument('--grad_norm', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # Output and logging
    parser.add_argument('--sweep_output_dir', type=str, default='./logs/marla_sweep',
                       help='Base output directory for sweep results')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='marla_sweep',
                       help='W&B project name')
    
    # Execution options
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel (requires job management)')
    parser.add_argument('--max_parallel', type=int, default=4,
                       help='Maximum number of parallel jobs (if --parallel)')
    
    # Other arguments
    parser.add_argument('--eval_freq', type=int, default=1,
                       help='Evaluation frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Checkpoint saving frequency')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.checkpoint is not None:
        if args.model is None:
            parser.error("--model is required when --checkpoint is provided")
        if args.num_classes is None:
            parser.error("--num_classes is required when --checkpoint is provided")
    
    if args.embedding_dir is not None:
        embedding_dir = Path(args.embedding_dir)
        if not embedding_dir.exists():
            parser.error(f"Embedding directory does not exist: {embedding_dir}")
    
    return args


def parse_list(list_str: str, dtype=float) -> List:
    """Parse comma-separated list string into list of values."""
    return [dtype(x.strip()) for x in list_str.split(',') if x.strip()]


def build_train_command(args, gamma: float, rank: int, output_dir: Path) -> List[str]:
    """Build command to run train.py for a specific gamma/rank combination."""
    cmd = [
        sys.executable,
        str(script_dir / 'train.py'),
        '--method', 'marla',
        '--dataset', args.dataset,
        '--data_dir', args.data_dir,
        '--gamma', str(gamma),
        '--spectral_rank', str(rank),
        '--num_epochs', str(args.num_epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--momentum', str(args.momentum),
        '--weight_decay', str(args.weight_decay),
        '--reg_coeff', str(args.reg_coeff),
        '--grad_norm', str(args.grad_norm),
        '--output_dir', str(output_dir),
        '--device', args.device,
        '--seed', str(args.seed),
        '--eval_freq', str(args.eval_freq),
        '--save_freq', str(args.save_freq),
    ]
    
    # Add mode-specific arguments
    if args.checkpoint is not None:
        cmd.extend([
            '--checkpoint', args.checkpoint,
            '--model', args.model,
            '--num_classes', str(args.num_classes),
            '--data_transform', args.data_transform,
        ])
    else:
        # Embedding-based mode
        embedding_dir = Path(args.embedding_dir)
        
        # Check if embeddings are in a single file (common case)
        single_embedding_file = embedding_dir / 'embeddings.pt'
        if single_embedding_file.exists():
            # Use simplified --embeddings argument
            cmd.extend(['--embeddings', str(single_embedding_file)])
        else:
            # Use separate files for each split (fallback to individual arguments)
            cmd.extend([
                '--train_embeddings', str(embedding_dir / 'train_embeddings.pt'),
                '--train_labels', str(embedding_dir / 'train_labels.pt'),
                '--val_embeddings', str(embedding_dir / 'val_embeddings.pt'),
                '--val_labels', str(embedding_dir / 'val_labels.pt'),
                '--test_embeddings', str(embedding_dir / 'test_embeddings.pt'),
                '--test_labels', str(embedding_dir / 'test_labels.pt'),
            ])
            
            # Add base weight/bias if they exist
            base_weight = embedding_dir / 'base_weight.pt'
            base_bias = embedding_dir / 'base_bias.pt'
            if base_weight.exists():
                cmd.extend(['--base_weight', str(base_weight)])
            if base_bias.exists():
                cmd.extend(['--base_bias', str(base_bias)])
            
            # Add groups if they exist
            for split in ['train', 'val', 'test']:
                groups_file = embedding_dir / f'{split}_groups.pt'
                if groups_file.exists():
                    cmd.extend([f'--{split}_groups', str(groups_file)])
    
    # Add wandb if requested
    if args.use_wandb:
        cmd.append('--use_wandb')
        cmd.extend(['--wandb_project', args.wandb_project])
        cmd.extend(['--wandb_run_name', f'gamma{gamma}_rank{rank}'])
    
    return cmd


def run_experiment(args, gamma: float, rank: int, output_dir: Path) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    print(f"\n{'='*80}")
    print(f"Running experiment: gamma={gamma}, rank={rank}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = build_train_command(args, gamma, rank, output_dir)
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        success = True
        error_msg = None
    except subprocess.CalledProcessError as e:
        success = False
        error_msg = str(e)
        print(f"ERROR: Experiment failed for gamma={gamma}, rank={rank}")
        print(f"Error: {error_msg}")
    
    elapsed_time = time.time() - start_time
    
    # Try to load results if available
    results_file = output_dir / 'results.json'
    metrics = {}
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load results from {results_file}: {e}")
    
    return {
        'gamma': gamma,
        'rank': rank,
        'success': success,
        'error': error_msg,
        'elapsed_time': elapsed_time,
        'output_dir': str(output_dir),
        'metrics': metrics,
    }


def collect_results(sweep_dir: Path) -> Dict[str, Any]:
    """Collect results from all completed experiments."""
    results = []
    
    for exp_dir in sweep_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Try to parse gamma and rank from directory name
        # Format: gamma{gamma}_rank{rank}
        try:
            parts = exp_dir.name.split('_')
            gamma = None
            rank = None
            for part in parts:
                if part.startswith('gamma'):
                    gamma = float(part.replace('gamma', ''))
                elif part.startswith('rank'):
                    rank = int(part.replace('rank', ''))
            
            if gamma is None or rank is None:
                continue
            
            # Load results if available
            results_file = exp_dir / 'results.json'
            metrics = {}
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        metrics = json.load(f)
                except Exception:
                    pass
            
            results.append({
                'gamma': gamma,
                'rank': rank,
                'output_dir': str(exp_dir),
                'metrics': metrics,
            })
        except Exception:
            continue
    
    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print summary of sweep results."""
    if not results:
        print("No results found.")
        return
    
    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    
    # Sort by gamma, then rank
    results_sorted = sorted(results, key=lambda x: (x['gamma'], x['rank']))
    
    # Print table header
    print(f"\n{'Gamma':<10} {'Rank':<8} {'Status':<12} {'Output Dir':<50}")
    print("-" * 80)
    
    for r in results_sorted:
        status = "Success" if r.get('success', True) else "Failed"
        output_dir = Path(r['output_dir']).name
        print(f"{r['gamma']:<10.2f} {r['rank']:<8} {status:<12} {output_dir:<50}")
    
    # Print best results if metrics available
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    
    results_with_metrics = [r for r in results_sorted if r.get('metrics')]
    if results_with_metrics:
        # Try to find best worst-group accuracy
        best_wga = None
        best_config = None
        
        for r in results_with_metrics:
            metrics = r['metrics']
            # Look for worst-group accuracy in various possible keys
            wga = None
            if 'test_worst_group_accuracy' in metrics:
                wga = metrics['test_worst_group_accuracy']
            elif 'worst_group_accuracy' in metrics:
                wga = metrics['worst_group_accuracy']
            elif 'test_metrics' in metrics and 'worst_group_accuracy' in metrics['test_metrics']:
                wga = metrics['test_metrics']['worst_group_accuracy']
            
            if wga is not None:
                if best_wga is None or wga > best_wga:
                    best_wga = wga
                    best_config = r
        
        if best_config:
            print(f"\nBest Worst-Group Accuracy: {best_wga:.4f}")
            print(f"  Gamma: {best_config['gamma']}")
            print(f"  Rank: {best_config['rank']}")
            print(f"  Output: {best_config['output_dir']}")


def main():
    args = parse_args()
    
    # Parse hyperparameter lists
    gamma_list = parse_list(args.gamma_list, dtype=float)
    rank_list = parse_list(args.rank_list, dtype=int)
    
    print(f"Hyperparameter Sweep Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Gamma values: {gamma_list}")
    print(f"  Rank values: {rank_list}")
    print(f"  Total experiments: {len(gamma_list) * len(rank_list)}")
    print(f"  Output directory: {args.sweep_output_dir}")
    
    # Create sweep output directory
    sweep_dir = Path(args.sweep_output_dir) / args.dataset
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sweep configuration
    config = {
        'dataset': args.dataset,
        'data_dir': args.data_dir,
        'gamma_list': gamma_list,
        'rank_list': rank_list,
        'checkpoint': args.checkpoint,
        'embedding_dir': args.embedding_dir,
        'model': args.model,
        'num_classes': args.num_classes,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'seed': args.seed,
    }
    config_file = sweep_dir / 'sweep_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved sweep configuration to: {config_file}")
    
    # Run experiments
    results = []
    total_experiments = len(gamma_list) * len(rank_list)
    experiment_num = 0
    
    for gamma, rank in product(gamma_list, rank_list):
        experiment_num += 1
        print(f"\n[{experiment_num}/{total_experiments}] Starting experiment...")
        
        # Create experiment-specific output directory
        exp_output_dir = sweep_dir / f'gamma{gamma}_rank{rank}'
        
        # Run experiment
        result = run_experiment(args, gamma, rank, exp_output_dir)
        results.append(result)
        
        # Save intermediate results
        results_file = sweep_dir / 'sweep_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print_summary(results)
    
    # Save final results
    results_file = sweep_dir / 'sweep_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sweep results to: {results_file}")
    
    print("\nSweep completed!")


if __name__ == '__main__':
    main()
