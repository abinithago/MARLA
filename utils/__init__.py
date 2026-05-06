"""
Utility modules for MARLA training and evaluation.
"""

from .metrics import (
    compute_mean_accuracy,
    compute_class_balanced_accuracy,
    compute_worst_group_accuracy,
    compute_worst_percentile_accuracy,
    compute_all_metrics,
)

from .logging import MARLALogger

from .training import (
    train_epoch,
    eval_model,
    AverageMeter,
)

from .embedding_extraction import (
    extract_embeddings,
    extract_embeddings_with_classifier,
    save_embeddings_dict,
)

__all__ = [
    'compute_mean_accuracy',
    'compute_class_balanced_accuracy',
    'compute_worst_group_accuracy',
    'compute_worst_percentile_accuracy',
    'compute_all_metrics',
    'MARLALogger',
    'train_epoch',
    'eval_model',
    'AverageMeter',
    'extract_embeddings',
    'extract_embeddings_with_classifier',
    'save_embeddings_dict',
]
