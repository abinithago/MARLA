"""
Logging infrastructure for MARLA training.

Provides logging for train/val/test splits with comprehensive metrics:
- Mean accuracy
- Class balanced accuracy
- Worst group accuracy (WGA)
- Worst 10th percentile accuracy
"""

import os
import logging
import pickle
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .metrics import compute_all_metrics


class MARLALogger:
    """
    Logger for MARLA training that tracks comprehensive metrics across train/val/test splits.
    """
    
    def __init__(
        self,
        output_dir: str,
        use_wandb: bool = False,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize MARLA logger.
        
        Args:
            output_dir: Directory to save logs and checkpoints
            use_wandb: Whether to use Weights & Biases logging
            project: W&B project name
            run_name: W&B run name
            config: Configuration dictionary to log
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Initialize logger
        self.logger = self._setup_logger()
        
        # Log configuration
        if config:
            self.log_config(config)
        
        # Initialize W&B if requested
        if self.use_wandb:
            wandb.init(
                dir=str(self.output_dir),
                project=project,
                name=run_name,
                config=config or {}
            )
        
        # Track best metrics
        self.best_metrics = {
            'val': {
                'mean_accuracy': 0.0,
                'class_balanced_accuracy': 0.0,
                'worst_group_accuracy': 0.0,
                'percentile_25_accuracy': 0.0,
                'logit_entropy': 0.0,
                'exp_weighted_accuracy': 0.0,
            },
            'test': {
                'mean_accuracy': 0.0,
                'class_balanced_accuracy': 0.0,
                'worst_group_accuracy': 0.0,
                'percentile_25_accuracy': 0.0,
                'logit_entropy': 0.0,
            }
        }
        
        # Store all metrics by epoch
        self.metrics = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup Python logger."""
        logger = logging.getLogger(str(self.output_dir))
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.output_dir / "info.log")
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        
        return logger
    
    def log_config(self, config: Dict):
        """Log configuration to file and W&B."""
        # Save to YAML
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        # Save to pickle
        with open(self.output_dir / 'config.pkl', 'wb') as f:
            pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Log to console
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_train_results_from_acc_groups(
        self,
        epoch: int,
        acc_groups: Dict[int, Any],  # Dict of {group_id: AverageMeter}
        split: str = 'train',
        n_spurious: int = 1
    ):
        """
        Log train results from acc_groups dictionary.
        
        Args:
            epoch: Current epoch number
            acc_groups: Dictionary of {group_id: AverageMeter} with accuracy per group
            split: Split name (default: 'train')
            n_spurious: Number of spurious attributes (for group encoding, default: 1)
        """
        from .training import AverageMeter
        
        # Convert acc_groups to format
        results = {}
        all_correct = 0
        all_total = 0
        
        # Get group accuracies and format like AFR: accuracy_y_s
        for group_id, acc_meter in acc_groups.items():
            if isinstance(acc_meter, AverageMeter):
                # Format: accuracy_y_s where y = group // n_spurious, s = group % n_spurious
                y = group_id // n_spurious
                s = group_id % n_spurious
                key = f"accuracy_{y}_{s}"
                results[key] = acc_meter.avg
                all_correct += acc_meter.sum
                all_total += acc_meter.count
        
        if all_total == 0:
            return
        
        mean_accuracy = all_correct / all_total if all_total > 0 else 0.0
        worst_accuracy = min(results.values()) if results else 0.0
        results["mean_accuracy"] = mean_accuracy
        results["worst_accuracy"] = worst_accuracy
        
        # Store metrics
        if epoch not in self.metrics:
            self.metrics[epoch] = {}
        
        # Log to console
        self.logger.info(f"{split} results")
        text = ""
        for key, val in results.items():
            text += f"{key}: {val:1.3e} | "
        self.logger.info(text)
        
        # Also store in our format for compatibility
        tagged_results = {f"{split}_{k}": v for k, v in results.items()}
        self.metrics[epoch].update(tagged_results)
    
    def log_metrics(
        self,
        epoch: int,
        logits: torch.Tensor,
        labels: torch.Tensor,
        split: str,
        groups: Optional[torch.Tensor] = None,
        exp_weight_factor: Optional[float] = None
    ):
        """
        Compute and log metrics for a given split.
        
        Args:
            epoch: Current epoch number
            logits: Model logits, shape (n, C)
            labels: True labels, shape (n,)
            split: Split name ('train', 'val', or 'test')
            groups: Group assignments, shape (n,) (optional)
            exp_weight_factor: Exponential weighting factor for validation metric (optional, used for selection only)
        """
        # Compute all metrics (including exp_weighted_accuracy if exp_weight_factor provided)
        metrics = compute_all_metrics(logits, labels, groups=groups, exp_weight_factor=exp_weight_factor if split == 'val' else None)
        
        # Store metrics
        if epoch not in self.metrics:
            self.metrics[epoch] = {}
        
        # Add split prefix to metric names
        split_metrics = {f"{split}_{k}": v for k, v in metrics.items()}
        self.metrics[epoch].update(split_metrics)
        
        # Log to console
        self.logger.info(f"\n{split.upper()} Metrics (Epoch {epoch}):")
        self.logger.info(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        self.logger.info(f"  Class Balanced Accuracy: {metrics['class_balanced_accuracy']:.4f}")
        if 'exp_weighted_accuracy' in metrics:
            self.logger.info(f"  Exponentially Weighted Accuracy: {metrics['exp_weighted_accuracy']:.4f}")
        
        if metrics['worst_group_accuracy'] is not None:
            self.logger.info(f"  Worst Group Accuracy (WGA): {metrics['worst_group_accuracy']:.4f}")
        else:
            self.logger.info(f"  Worst Group Accuracy (WGA): N/A (no groups provided)")
        
        
        # Log per-class accuracies
        if metrics['per_class_accuracies']:
            self.logger.info("  Per-Class Accuracies:")
            for class_id, acc in sorted(metrics['per_class_accuracies'].items()):
                self.logger.info(f"    Class {class_id}: {acc:.4f}")
        
        # Log per-group accuracies if available
        if metrics['group_accuracies'] is not None:
            self.logger.info("  Per-Group Accuracies:")
            for group_id, acc in sorted(metrics['group_accuracies'].items()):
                self.logger.info(f"    Group {group_id}: {acc:.4f}")
        
        # Update best metrics
        if split in ['val', 'test']:
            self._update_best_metrics(split, metrics, epoch)
        
        # Log to W&B
        if self.use_wandb:
            wandb_metrics = {f"{split}/{k}": v for k, v in metrics.items() if v is not None}
            wandb_metrics['epoch'] = epoch
            wandb.log(wandb_metrics, step=epoch)
    
    def _update_best_metrics(self, split: str, metrics: Dict, epoch: int):
        """Update best metrics and log improvements."""
        improved = False
        
        for metric_name in ['mean_accuracy', 'class_balanced_accuracy', 
                           'worst_group_accuracy', 'percentile_25_accuracy', 'logit_entropy', 'exp_weighted_accuracy']:
            # Skip if metric doesn't exist in metrics dict or is None
            if metric_name not in metrics or metrics[metric_name] is None:
                continue
            
            current = metrics[metric_name]
            best = self.best_metrics[split][metric_name]
            
            if current > best:
                self.best_metrics[split][metric_name] = current
                improved = True
                if metric_name == 'percentile_25_accuracy':
                    self.logger.info(
                        f"New best {split} worst_25th_percentile_accuracy: {current:1.3e} (epoch {epoch})"
                    )
                else:
                    self.logger.info(
                        f"New best {split} {metric_name}: {current:.4f} (epoch {epoch})"
                    )
                
                # Log to W&B
                if self.use_wandb:
                    wandb.log({f"best_{split}/{metric_name}": current}, step=epoch)
        
        # Store best metrics in epoch metrics
        if epoch not in self.metrics:
            self.metrics[epoch] = {}
        
        # Store epoch number for later identification of best epoch
        self.metrics[epoch]['epoch'] = epoch
        
        for metric_name, best_value in self.best_metrics[split].items():
            self.metrics[epoch][f"best_{split}_{metric_name}"] = best_value
    
    def save_checkpoint(self, model: torch.nn.Module, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': self.metrics.get(epoch, {}),
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Save as latest
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
        
        # Save as best if applicable
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')
            self.logger.info(f"Saved best checkpoint at epoch {epoch}")
    
    def save_metrics(self):
        """Save all metrics to file."""
        metrics_path = self.output_dir / 'metrics.pkl'
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Also save as JSON-readable format (convert tensors to lists)
        metrics_dict = {}
        for epoch, epoch_metrics in self.metrics.items():
            metrics_dict[epoch] = {}
            for key, value in epoch_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics_dict[epoch][key] = value.tolist() if value.numel() > 1 else value.item()
                elif isinstance(value, dict):
                    metrics_dict[epoch][key] = {
                        k: (v.tolist() if isinstance(v, torch.Tensor) and v.numel() > 1 
                            else v.item() if isinstance(v, torch.Tensor) 
                            else v)
                        for k, v in value.items()
                    }
                else:
                    metrics_dict[epoch][key] = value
        
        import json
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        self.logger.info(f"Saved metrics to {metrics_path}")
    
    def finalize(self):
        """Finalize logging (save metrics, close W&B, etc.)."""
        self.save_metrics()
        
        # Print summary
        self.logger.info("\n" + "="*50)
        self.logger.info("Training Summary")
        self.logger.info("="*50)
        
        for split in ['val', 'test']:
            self.logger.info(f"\nBest {split.upper()} Metrics:")
            for metric_name, best_value in self.best_metrics[split].items():
                self.logger.info(f"  {metric_name}: {best_value:.4f}")
        
        if self.use_wandb:
            wandb.finish()
        
        self.logger.info(f"\nLogs saved to: {self.output_dir}")
