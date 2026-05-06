#!/usr/bin/env python3
"""
Configuration for MARLA (Misclassification Aware Rank-Limited Adaptation)

This module provides paths and configuration for datasets and analysis.

Note: Metadata files (metadata.csv) are automatically generated from raw datasets.
You only need to point to your dataset directories - no manual metadata formatting required!
"""

import os
from pathlib import Path

# Base directory for MARLA
MARLA_DIR = Path(__file__).parent.absolute()
DATA_DIR = MARLA_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"

# Default conda environment
DEFAULT_CONDA_ENV = "marla_env"

# Update these paths to point to your dataset directories
# Metadata will be automatically generated from raw data when you first use a dataset
DATASET_PATHS = {
    "celeba": DATASETS_DIR / "celeba",  # Path to CelebA dataset directory
    "civilcomments": DATASETS_DIR / "civilcomments",  # Path to CivilComments dataset directory
    "multinli": DATASETS_DIR / "multinli",  # Path to MultiNLI dataset directory
    "waterbirds": DATASETS_DIR / "waterbirds",  # Path to Waterbirds dataset directory
}

# Dataset-specific configuration
DATASET_CONFIG = {
    "celeba": {
        "name": "CelebA",
        "type": "image",
        "metadata_file": "list_attr_celeba.txt",  # Standard CelebA metadata
        "split_file": "list_eval_partition.txt",  # Train/val/test splits
        "data_dir_key": "data_dir",
    },
    "civilcomments": {
        "name": "CivilComments",
        "type": "text",
        "framework": "wilds",  # Uses WILDS package
        "data_dir_key": "data_dir",  # Path to civilcomments directory
        "wilds_root_dir": None,  # If using WILDS, parent of data_dir is root_dir
        "dataset_name": "civilcomments",
    },
    "multinli": {
        "name": "MultiNLI",
        "type": "text",
        "metadata_file": "metadata_random.csv",  # Custom metadata with splits
        "root_dir_key": "root_dir",
    },
    "waterbirds": {
        "name": "Waterbirds",
        "type": "image",
        "metadata_file": "metadata.csv",  # Generated metadata
        "data_dir_key": "data_dir",
    },
}

def get_dataset_path(dataset_name):
    """Get the path to a dataset."""
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_PATHS.keys())}")
    path = DATASET_PATHS[dataset_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {path}\n"
            f"Please update DATASET_PATHS in config.py to point to your dataset directory.\n"
            f"Metadata will be automatically generated when you first use the dataset."
        )
    return path

def get_dataset_config(dataset_name):
    """Get configuration for a dataset."""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_CONFIG[dataset_name]
