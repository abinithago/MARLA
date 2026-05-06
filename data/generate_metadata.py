"""
Metadata generation utilities for MARLA datasets.

Automatically generates metadata.csv files from raw dataset directories.
This allows users to simply point to their dataset directories without
manually formatting metadata files.
"""

import os
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)


def generate_metadata_celeba(data_dir: str, output_path: Optional[str] = None) -> str:
    """
    Generate metadata.csv for CelebA dataset.
    
    Expects:
    - list_eval_partition.txt (train/val/test splits)
    - list_attr_celeba.txt (attributes)
    - img_align_celeba/ directory with images
    
    Args:
        data_dir: Path to CelebA dataset directory
        output_path: Optional path to save metadata.csv (default: data_dir/metadata.csv)
    
    Returns:
        Path to generated metadata file
    """
    logging.info(f"Generating metadata for CelebA from {data_dir}...")
    
    data_path = Path(data_dir)
    split_file = data_path / "list_eval_partition.txt"
    attr_file = data_path / "list_attr_celeba.txt"
    
    if not split_file.exists():
        raise FileNotFoundError(
            f"CelebA split file not found: {split_file}\n"
            f"Expected list_eval_partition.txt in {data_dir}"
        )
    if not attr_file.exists():
        raise FileNotFoundError(
            f"CelebA attributes file not found: {attr_file}\n"
            f"Expected list_attr_celeba.txt in {data_dir}"
        )
    
    # Read splits (format: filename split_id)
    with open(split_file, "r") as f:
        splits = f.readlines()
    
    # Read attributes (skip first 2 lines, format: filename attr1 attr2 ...)
    with open(attr_file, "r") as f:
        attrs = f.readlines()[2:]
    
    if len(splits) != len(attrs):
        raise ValueError(f"Mismatch: {len(splits)} splits but {len(attrs)} attribute lines")
    
    # Generate metadata
    metadata_rows = []
    for i, (split_line, attr_line) in enumerate(zip(splits, attrs)):
        filename, split_id = split_line.strip().split()
        attr_values = attr_line.strip().split()[1:]  # Skip filename
        
        # CelebA: y = Blond_Hair (attr index 9), a = Male (attr index 20)
        # 1 = has attribute, -1 = doesn't have attribute
        y = 1 if attr_values[9] == "1" else 0  # Blond_Hair
        a = 1 if attr_values[20] == "1" else 0  # Male
        
        # Split: 0=train, 1=val, 2=test
        split = int(split_id)
        
        metadata_rows.append({
            "id": i + 1,
            "filename": filename,
            "split": split,
            "y": y,
            "a": a
        })
    
    df = pd.DataFrame(metadata_rows)
    
    # Save metadata
    if output_path is None:
        output_path = data_path / "metadata.csv"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved metadata to {output_path} ({len(df)} samples)")
    
    return str(output_path)


def generate_metadata_waterbirds(data_dir: str, output_path: Optional[str] = None) -> str:
    """
    Generate metadata.csv for Waterbirds dataset.
    
    Expects:
    - waterbird_complete95_forest2water2/metadata.csv (original metadata)
    
    Args:
        data_dir: Path to Waterbirds dataset directory
        output_path: Optional path to save metadata.csv (default: data_dir/metadata.csv)
    
    Returns:
        Path to generated metadata file
    """
    logging.info(f"Generating metadata for Waterbirds from {data_dir}...")
    
    data_path = Path(data_dir)
    
    # Look for original metadata in subdirectory
    possible_metadata_paths = [
        data_path / "waterbird_complete95_forest2water2" / "metadata.csv",
        data_path / "metadata.csv",  # Already exists
    ]
    
    original_metadata = None
    for path in possible_metadata_paths:
        if path.exists():
            original_metadata = path
            break
    
    if original_metadata is None:
        raise FileNotFoundError(
            f"Waterbirds metadata not found. Tried:\n"
            f"  - {possible_metadata_paths[0]}\n"
            f"  - {possible_metadata_paths[1]}\n"
            f"Please ensure metadata.csv exists in one of these locations."
        )
    
    # Read and reformat metadata
    df = pd.read_csv(original_metadata)
    
    # Rename columns to match expected format
    if "img_id" in df.columns:
        df = df.rename(columns={"img_id": "id"})
    if "img_filename" in df.columns:
        df = df.rename(columns={"img_filename": "filename"})
    if "place" in df.columns:
        df = df.rename(columns={"place": "a"})
    
    # Ensure required columns exist
    required_cols = ["id", "filename", "split", "y", "a"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in metadata: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )
    
    # Select and reorder columns
    df = df[required_cols].copy()
    
    # Save metadata
    if output_path is None:
        output_path = data_path / "metadata.csv"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved metadata to {output_path} ({len(df)} samples)")
    
    return str(output_path)


def generate_metadata_civilcomments(data_dir: str, output_path: Optional[str] = None, granularity: str = "coarse") -> str:
    """
    Generate metadata.csv for CivilComments dataset.
    
    Expects:
    - all_data_with_identities.csv (WILDS format)
    
    Args:
        data_dir: Path to CivilComments dataset directory
        output_path: Optional path to save metadata.csv
        granularity: "coarse" or "fine" (default: "coarse")
    
    Returns:
        Path to generated metadata file
    """
    logging.info(f"Generating metadata for CivilComments from {data_dir}...")
    
    data_path = Path(data_dir)
    data_file = data_path / "all_data_with_identities.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"CivilComments data file not found: {data_file}\n"
            f"Expected all_data_with_identities.csv in {data_dir}"
        )
    
    # Read data
    df = pd.read_csv(data_file, index_col=0)
    
    # Group attributes
    group_attrs = [
        "male", "female", "LGBTQ",
        "christian", "muslim", "other_religions",
        "black", "white",
    ]
    
    # Check which attributes exist
    available_attrs = [attr for attr in group_attrs if attr in df.columns]
    if not available_attrs:
        raise ValueError(
            f"No group attributes found in {data_file}.\n"
            f"Expected columns: {group_attrs}\n"
            f"Found columns: {list(df.columns)}"
        )
    
    # Prepare data
    cols_to_keep = ["comment_text", "split", "toxicity"] + available_attrs
    df = df[cols_to_keep].copy()
    df = df.rename(columns={"toxicity": "y"})
    df["y"] = (df["y"] >= 0.5).astype(int)
    df[available_attrs] = (df[available_attrs] >= 0.5).astype(int)
    
    # Add "no active attributes" group
    df["no_active_attributes"] = 0
    df.loc[df[available_attrs].sum(axis=1) == 0, "no_active_attributes"] = 1
    
    # Generate metadata based on granularity
    if granularity == "coarse":
        # Train: use "no active attributes" as group
        # Val/Test: use individual attributes
        few_groups = []
        train_df = df[df["split"] == "train"].copy()
        train_df = train_df.rename(columns={"no_active_attributes": "a"})
        few_groups.append(train_df[["y", "split", "comment_text", "a"]])
        
        for split, split_df in df.groupby("split"):
            if split != "train":
                for i, attr in enumerate(available_attrs):
                    test_df = split_df.loc[
                        split_df[attr] == 1, ["y", "split", "comment_text"]
                    ].copy()
                    test_df["a"] = i
                    few_groups.append(test_df)
        
        metadata_df = pd.concat(few_groups).reset_index(drop=True)
    else:  # fine
        # All splits: use individual attributes
        all_groups = []
        for split, split_df in df.groupby("split"):
            for i, attr in enumerate(available_attrs):
                test_df = split_df.loc[
                    split_df[attr] == 1, ["y", "split", "comment_text"]
                ].copy()
                test_df["a"] = i
                all_groups.append(test_df)
        
        metadata_df = pd.concat(all_groups).reset_index(drop=True)
    
    # Format metadata
    metadata_df.index.name = "filename"
    metadata_df = metadata_df.reset_index()
    metadata_df["id"] = metadata_df["filename"]
    metadata_df["split"] = metadata_df["split"].replace({"train": 0, "val": 1, "test": 2})
    text = metadata_df.pop("comment_text")
    
    # Save metadata
    if output_path is None:
        output_path = data_path / f"metadata_civilcomments_{granularity}.csv"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df[["id", "filename", "split", "y", "a"]].to_csv(output_path, index=False)
    
    # Save text separately
    text_path = output_path.parent / f"civilcomments_{granularity}.csv"
    text.to_csv(text_path, index=False)
    
    logging.info(f"Saved metadata to {output_path} ({len(metadata_df)} samples)")
    
    return str(output_path)


def generate_metadata_multinli(data_dir: str, output_path: Optional[str] = None) -> str:
    """
    Generate metadata.csv for MultiNLI dataset.
    
    Expects:
    - data/metadata_random.csv (or metadata_random.csv in root)
    
    Args:
        data_dir: Path to MultiNLI dataset directory
        output_path: Optional path to save metadata.csv
    
    Returns:
        Path to generated metadata file
    """
    logging.info(f"Generating metadata for MultiNLI from {data_dir}...")
    
    data_path = Path(data_dir)
    
    # Look for metadata in common locations
    possible_metadata_paths = [
        data_path / "data" / "metadata_random.csv",
        data_path / "metadata_random.csv",
    ]
    
    original_metadata = None
    for path in possible_metadata_paths:
        if path.exists():
            original_metadata = path
            break
    
    if original_metadata is None:
        raise FileNotFoundError(
            f"MultiNLI metadata not found. Tried:\n"
            f"  - {possible_metadata_paths[0]}\n"
            f"  - {possible_metadata_paths[1]}\n"
            f"Please ensure metadata_random.csv exists in one of these locations."
        )
    
    # Read and reformat metadata
    df = pd.read_csv(original_metadata, index_col=0)
    
    # Rename columns
    if "gold_label" in df.columns:
        df = df.rename(columns={"gold_label": "y"})
    if "sentence2_has_negation" in df.columns:
        df = df.rename(columns={"sentence2_has_negation": "a"})
    
    # Map labels if needed
    if df["y"].dtype == object:
        label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
        df["y"] = df["y"].map(label_map)
    
    # Ensure split is numeric
    if df["split"].dtype == object:
        split_map = {"train": 0, "val": 1, "test": 2}
        df["split"] = df["split"].map(split_map)
    
    # Format metadata
    df = df.reset_index(drop=True)
    df.index.name = "id"
    df = df.reset_index()
    df["filename"] = df["id"]
    
    # Select required columns
    required_cols = ["id", "filename", "split", "y", "a"]
    df = df[required_cols].copy()
    
    # Save metadata
    if output_path is None:
        output_path = data_path / "metadata.csv"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved metadata to {output_path} ({len(df)} samples)")
    
    return str(output_path)


def generate_metadata(dataset_name: str, data_dir: str, output_path: Optional[str] = None, **kwargs) -> str:
    """
    Generate metadata for a dataset.
    
    Args:
        dataset_name: Name of dataset ('celeba', 'waterbirds', 'civilcomments', 'multinli')
        data_dir: Path to dataset directory
        output_path: Optional path to save metadata.csv
        **kwargs: Additional arguments for specific dataset generators
    
    Returns:
        Path to generated metadata file
    """
    generators = {
        "celeba": generate_metadata_celeba,
        "waterbirds": generate_metadata_waterbirds,
        "civilcomments": generate_metadata_civilcomments,
        "multinli": generate_metadata_multinli,
    }
    
    if dataset_name not in generators:
        raise ValueError(
            f"Unknown dataset: {dataset_name}\n"
            f"Available: {list(generators.keys())}"
        )
    
    return generators[dataset_name](data_dir, output_path, **kwargs)


if __name__ == '__main__':
    """CLI entry point for metadata generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate metadata.csv for MARLA datasets from raw data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['celeba', 'waterbirds', 'civilcomments', 'multinli'],
        help='Dataset name'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Optional output path for metadata.csv (default: data_dir/metadata.csv)'
    )
    parser.add_argument(
        '--granularity',
        type=str,
        default='coarse',
        choices=['coarse', 'fine'],
        help='Granularity for CivilComments (coarse or fine)'
    )
    
    args = parser.parse_args()
    
    kwargs = {}
    if args.dataset == 'civilcomments':
        kwargs['granularity'] = args.granularity
    
    output_path = generate_metadata(
        args.dataset,
        args.data_dir,
        args.output_path,
        **kwargs
    )
    
    print(f"✓ Metadata generated successfully: {output_path}")
