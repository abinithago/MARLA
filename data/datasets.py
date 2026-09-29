"""
Dataset classes for MARLA datasets.

Provides dataset loaders for:
- CelebA and Waterbirds: SpuriousDataset
- CivilComments: WILDS wrapper
- MultiNLI: MultiNLIDataset

All datasets automatically generate metadata.csv from raw data if it doesn't exist.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from typing import Optional, Callable

from .generate_metadata import generate_metadata

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _bincount_array_as_tensor(arr):
    """Convert numpy bincount to tensor."""
    return torch.from_numpy(np.bincount(arr)).long()


def _randomly_split_in_two(samples, prop=0.8, seed=21, max_prop=1.0):
    """Randomly split samples based on proportion."""
    random.seed(seed)
    random.shuffle(samples)
    samples = samples[:int(len(samples) * max_prop)]
    first_total = int(len(samples) * np.abs(prop))
    if prop >= 0:
        return samples[:first_total]
    else:
        return samples[-first_total:]


def _get_split(split):
    """Convert split string to integer."""
    try:
        return ["train", "val", "test"].index(split)
    except ValueError:
        raise ValueError(f"Unknown split {split}")


def _cast_int(arr):
    """Cast array to int."""
    if isinstance(arr, np.ndarray):
        return arr.astype(int)
    elif isinstance(arr, torch.Tensor):
        return arr.int()
    else:
        raise NotImplementedError(f"Cannot cast {type(arr)} to int")


class SpuriousDataset(Dataset):
    """
    Dataset for CelebA and Waterbirds with spurious correlations.
    
    Loads images from metadata CSV files and handles train/val/test splits.
    Returns (image, label, group) tuples.
    """
    
    def __init__(
        self,
        basedir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        prop: float = 1.0,
        seed: int = 21,
        max_prop: float = 1.0
    ):
        """
        Initialize SpuriousDataset.
        
        Args:
            basedir: Base directory containing metadata.csv and images
            split: Split name ('train', 'val', or 'test')
            transform: Image transform to apply
            prop: Proportion of data to use (for subsampling)
            seed: Random seed for subsampling
            max_prop: Maximum proportion of data to consider
        """
        self.basedir = basedir
        self.transform = transform

        self.metadata_df = self._get_metadata(split)
        indices = np.arange(len(self.metadata_df))
        ind = _randomly_split_in_two(indices, prop=prop, seed=seed, max_prop=max_prop)
        self.metadata_df = self.metadata_df.iloc[np.sort(ind)]

        self.y_array = self.metadata_df["y"].values
        self.spurious_array = self.metadata_df["place"].values
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()
        self.filename_array = self.metadata_df["img_filename"].values

    def _get_metadata(self, split):
        """Load metadata for specified split. Auto-generates if missing."""
        split_i = _get_split(split)
        metadata_path = os.path.join(self.basedir, "metadata.csv")
        
        if not os.path.exists(metadata_path):
            import logging
            logging.info(f"Metadata not found at {metadata_path}. Attempting to auto-generate...")
            
            if os.path.exists(os.path.join(self.basedir, "list_attr_celeba.txt")):
                generate_metadata("celeba", self.basedir, metadata_path)
            elif os.path.exists(os.path.join(self.basedir, "waterbird_complete95_forest2water2", "metadata.csv")):
                generate_metadata("waterbirds", self.basedir, metadata_path)
            else:
                raise FileNotFoundError(
                    f"Metadata file not found: {metadata_path}\n"
                    f"Could not auto-generate metadata. Please ensure:\n"
                    f"  - For CelebA: list_attr_celeba.txt and list_eval_partition.txt exist\n"
                    f"  - For Waterbirds: waterbird_complete95_forest2water2/metadata.csv exists\n"
                    f"  - Or manually create metadata.csv with columns: id,filename,split,y,a"
                )
        
        metadata_df = pd.read_csv(metadata_path)
        metadata_df = metadata_df[metadata_df["split"] == split_i]
        return metadata_df

    def _count_attributes(self):
        """Count classes and spurious attributes."""
        self.n_classes = np.unique(self.y_array).size
        self.n_spurious = np.unique(self.spurious_array).size
        self.y_counts = _bincount_array_as_tensor(self.y_array)
        self.spurious_counts = _bincount_array_as_tensor(self.spurious_array)

    def _count_groups(self):
        """Count groups (combinations of class and spurious attribute)."""
        self.group_counts = _bincount_array_as_tensor(self.group_array)
        self.n_groups = len(self.group_counts)
        self.active_groups = np.unique(self.group_array).tolist()

    def _get_class_spurious_groups(self):
        """Compute group array from class and spurious attribute."""
        self.group_array = _cast_int(self.y_array * self.n_spurious + self.spurious_array)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Returns:
            image: Transformed image tensor
            label: Class label (int)
            group: Group ID (int)
        """
        y = int(self.y_array[idx])
        group = int(self.group_array[idx])
        x = self._image_getitem(idx)
        return x, y, group

    def _image_getitem(self, idx):
        """Load and transform image."""
        img_filename = self.filename_array[idx]
        
        # For CelebA, images are in img_align_celeba subdirectory
        # For Waterbirds, images are directly in basedir or subdirectories
        img_align_dir = os.path.join(self.basedir, "img_align_celeba")
        if os.path.exists(img_align_dir):
            img_path = os.path.join(img_align_dir, img_filename)
        else:
            # Try direct path or look for images in subdirectories
            img_path = os.path.join(self.basedir, img_filename)
            if not os.path.exists(img_path):
                # Waterbirds might have images in subdirectories
                # Try common patterns
                for subdir in ['images', 'data']:
                    alt_path = os.path.join(self.basedir, subdir, img_filename)
                    if os.path.exists(alt_path):
                        img_path = alt_path
                        break
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class MultiNLIDataset(Dataset):
    """
    Dataset for MultiNLI with pre-computed BERT features.
    
    Loads BERT tokenized features and metadata.
    Returns (input_ids, attention_mask, token_type_ids, label, group) tuples.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        metadata_file: str = "metadata_random.csv"
    ):
        """
        Initialize MultiNLIDataset.
        
        Args:
            root_dir: Root directory containing MultiNLI data
            split: Split name ('train', 'val', or 'test')
            transform: Optional transform (usually None for pre-tokenized data)
            metadata_file: Name of metadata CSV file
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load metadata (auto-generate if missing)
        metadata_path = os.path.join(root_dir, "data", metadata_file)
        if not os.path.exists(metadata_path):
            # Try alternative location
            metadata_path = os.path.join(root_dir, metadata_file)
        
        if not os.path.exists(metadata_path):
            # Auto-generate metadata
            import logging
            logging.info(f"Metadata not found at {metadata_path}. Attempting to auto-generate...")
            generate_metadata("multinli", root_dir, metadata_path)
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Could not auto-generate metadata. Please ensure:\n"
                f"  - data/metadata_random.csv exists in {root_dir}\n"
                f"  - Or manually create metadata.csv with columns: id,filename,split,y,a"
            )
        
        self.metadata_df = pd.read_csv(metadata_path)
        split_i = _get_split(split)
        self.metadata_df = self.metadata_df[self.metadata_df["split"] == split_i]
        
        # Load BERT features
        self._load_bert_features()
        
        # Get labels and groups from metadata
        self._process_metadata()
    
    def _load_bert_features(self):
        """Load pre-computed BERT features."""
        glue_data_dir = os.path.join(self.root_dir, "glue_data", "MNLI")
        
        # Load cached BERT features
        feature_files = [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]
        
        self.features_array = []
        for feature_file in feature_files:
            feature_path = os.path.join(glue_data_dir, feature_file)
            if os.path.exists(feature_path):
                features = torch.load(feature_path, map_location='cpu')
                self.features_array.extend(features)
        
        if len(self.features_array) == 0:
            raise FileNotFoundError(
                f"No BERT features found in {glue_data_dir}\n"
                f"Expected cached_*_bert-base-uncased_128_mnli files"
            )
        
        # Stack features
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in self.features_array], dtype=torch.long
        )
        self.all_input_masks = torch.tensor(
            [f.input_mask for f in self.features_array], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in self.features_array], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_id for f in self.features_array], dtype=torch.long
        )
        
        # Stack as (batch, seq_len, 3) tensor: [input_ids, attention_mask, token_type_ids]
        self.x_array = torch.stack(
            (self.all_input_ids, self.all_input_masks, self.all_segment_ids), dim=2
        )
    
    def _process_metadata(self):
        """Process metadata to get labels and groups."""
        # Map metadata indices to feature indices
        # This assumes metadata and features are aligned
        # You may need to adjust this based on your data structure
        
        # Get labels from metadata (gold_label column)
        if 'gold_label' in self.metadata_df.columns:
            label_dict = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
            self.metadata_df['y'] = self.metadata_df['gold_label'].map(label_dict)
        else:
            # Use label_ids from features if metadata doesn't have labels
            self.metadata_df['y'] = self.all_label_ids[:len(self.metadata_df)].numpy()
        
        # Get groups from metadata (sentence2_has_negation or similar)
        if 'sentence2_has_negation' in self.metadata_df.columns:
            self.metadata_df['group'] = self.metadata_df['sentence2_has_negation']
        elif 'a' in self.metadata_df.columns:
            self.metadata_df['group'] = self.metadata_df['a']
        else:
            # Default: use label as group
            self.metadata_df['group'] = self.metadata_df['y']
        
        self.y_array = self.metadata_df['y'].values
        self.group_array = self.metadata_df['group'].values
        
        # Count attributes
        self.n_classes = len(np.unique(self.y_array))
        self.n_groups = len(np.unique(self.group_array))
    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Returns:
            x: Stacked tensor of (input_ids, attention_mask, token_type_ids), shape (seq_len, 3)
            label: Class label (int)
            group: Group ID (int)
        """
        # Get features (stacked as [input_ids, attention_mask, token_type_ids])
        x = self.x_array[idx]
        
        # Get label and group
        y = int(self.y_array[idx])
        group = int(self.group_array[idx])
        
        return x, y, group


class CivilCommentsWILDSDataset(Dataset):
    """
    Wrapper for WILDS CivilComments dataset.
    
    Adapts WILDS dataset to match our interface.
    Returns (text, label, group) tuples.
    """
    
    def __init__(
        self,
        wilds_dataset,
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        """
        Initialize CivilCommentsWILDSDataset.
        
        Args:
            wilds_dataset: WILDS dataset object (from wilds.get_dataset)
            split: Split name ('train', 'val', or 'test')
            transform: Text transform (tokenizer)
        """
        self.wilds_dataset = wilds_dataset
        self.split = split
        self.transform = transform
        
        # Get subset
        self.subset = wilds_dataset.get_subset(split, transform=transform)
        
        # Extract metadata for groups
        # WILDS uses metadata array where each row contains group information
        self.metadata = self.subset.metadata_array
        
        # Get labels
        self.y_array = self.subset.y_array
        
        # Extract groups from metadata (toxic/non-toxic is typically in first column)
        if self.metadata.shape[1] > 0:
            self.group_array = self.metadata[:, 0].numpy()  # Toxic/non-toxic
        else:
            self.group_array = np.zeros(len(self.y_array), dtype=int)
        
        # Count attributes
        self.n_classes = len(np.unique(self.y_array))
        self.n_groups = len(np.unique(self.group_array))
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Returns:
            x: Transformed text (tokenized if transform provided)
            label: Class label (int)
            group: Group ID (int)
        """
        # Get item from WILDS subset
        item = self.subset[idx]
        
        # WILDS returns (x, y, metadata) or just (x, y)
        if len(item) >= 2:
            x = item[0]
            y = int(item[1])
        else:
            x = item
            y = int(self.y_array[idx])
        
        # Get group
        group = int(self.group_array[idx])
        
        return x, y, group
