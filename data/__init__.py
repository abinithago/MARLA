"""
Data utilities for MARLA datasets.

Provides transforms and data loading utilities for:
- CelebA and Waterbirds (image datasets)
- CivilComments and MultiNLI (text datasets)
"""

from .transforms import (
    AugWaterbirdsCelebATransform,
    NoAugWaterbirdsCelebATransform,
    NoAugNoNormWaterbirdsCelebATransform,
    ImageNetAugmixTransform,
    BertTokenizeTransform,
    BertTransform,
    get_transform,
    IMAGENET_STATS,
)

from .datasets import (
    SpuriousDataset,
    MultiNLIDataset,
    CivilCommentsWILDSDataset,
)

from .generate_metadata import (
    generate_metadata,
    generate_metadata_celeba,
    generate_metadata_waterbirds,
    generate_metadata_civilcomments,
    generate_metadata_multinli,
)

__all__ = [
    # Transforms
    'AugWaterbirdsCelebATransform',
    'NoAugWaterbirdsCelebATransform',
    'NoAugNoNormWaterbirdsCelebATransform',
    'ImageNetAugmixTransform',
    'BertTokenizeTransform',
    'BertTransform',
    'get_transform',
    'IMAGENET_STATS',
    # Datasets
    'SpuriousDataset',
    'MultiNLIDataset',
    'CivilCommentsWILDSDataset',
    # Metadata generation
    'generate_metadata',
    'generate_metadata_celeba',
    'generate_metadata_waterbirds',
    'generate_metadata_civilcomments',
    'generate_metadata_multinli',
]
