# MARLA: Misclassification Aware Rank-Limited Adaptation

MARLA (Misclassification Aware Rank-Limited Adaptation) is a method for learning low-rank adjustments to base classifiers by identifying error-informed subspaces and learning adjustments within those subspaces.

## Installation

### Setup

1. Clone or download this directory **into a folder named `marla`**. The training scripts add the repository’s parent directory to `sys.path` and import the code as the `marla` package (`from marla.models import …`), so the checkout folder name must be `marla`.

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate marla_env
```

The `environment.yml` file includes all required dependencies:
- Python 3.9.7
- PyTorch 1.13.0 and torchvision 0.14.0
- transformers 4.24.0 (for BERT models)
- wilds 2.0.0 (for CivilComments dataset)
- numpy, pandas, scikit-learn, scipy
- Other utilities (tqdm, pyyaml, wandb, matplotlib, etc.)

3. **Set up your datasets**:
   - Point to your dataset directories (see Dataset Setup below)
   - **Metadata is automatically generated** from raw data - no manual formatting needed!

## Directory Structure

```
marla/                         # Package root
├── data/                     # Dataset loaders and transforms
│   ├── datasets.py           # Dataset classes (SpuriousDataset, MultiNLIDataset, etc.)
│   ├── transforms.py         # Data augmentation transforms
│   ├── augmix.py             # AugMix augmentation
│   └── generate_metadata.py  # Auto-generate metadata from raw datasets
├── losses/                   # Loss functions
│   └── __init__.py           # MARLA weights, spectral decomposition, loss functions
├── models/                   # Model architectures
│   ├── __init__.py           # MARLAModel class
│   └── architectures.py      # ResNet50, BERT model factories
├── optimizers/               # Optimizer and scheduler factories
├── utils/                    # Utilities (metrics, logging, training, embedding extraction)
├── scripts/                  # Training and extraction scripts
│   ├── train_stage1.py       # Stage 1: Train ERM base classifier
│   ├── extract_embeddings.py # Extract embeddings from Stage 1 model
│   └── train.py              # Stage 2: Train MARLA/ERM on embeddings
├── config.py                 # Configuration and dataset paths
├── generate_metadata.py      # CLI tool for manual metadata generation
├── environment.yml           # Conda environment definition
├── README.md                 # This file
```

## Usage

MARLA follows a two-stage training paradigm:

### Stage 1: Train ERM Base Classifier

Train a standard ERM classifier on your dataset:

```bash
python scripts/train_stage1.py \
    --dataset waterbirds \
    --data_dir /path/to/waterbirds \
    --model imagenet_resnet50_pretrained \
    --num_epochs 50 \
    --batch_size 32 \
    --lr 0.003 \
    --output_dir outputs/ERM/waterbirds/stage1_seed1
```

**Note**: `--data_dir` should point to your raw dataset directory. Metadata will be auto-generated if missing.

### Stage 2: Extract Embeddings

Extract embeddings from the trained Stage 1 model:

```bash
python scripts/extract_embeddings.py \
    --checkpoint outputs/ERM/waterbirds/stage1_seed1/checkpoint.pt \
    --model imagenet_resnet50_pretrained \
    --num_classes 2 \
    --dataset waterbirds \
    --data_dir /path/to/waterbirds \
    --output_path outputs/ERM/waterbirds/stage1_seed1/embeddings.pt
```

### Stage 3: Train MARLA

Train MARLA on the extracted embeddings:

```bash
python scripts/train.py \
    --method marla \
    --embeddings outputs/ERM/waterbirds/stage1_seed1/embeddings.pt \
    --spectral_rank 1 \
    --gamma 10.0 \
    --num_epochs 1000 \
    --lr 0.01 \
    --output_dir outputs/MARLA/waterbirds/stage1_seed1
```

**Note**: The `--embeddings` argument automatically loads all data (train/val/test embeddings, labels, groups, and base classifier weights) from a single `.pt` file. If you need to use separate files for different splits, you can still use the individual `--train_embeddings`, `--val_embeddings`, etc. arguments.

## Hyperparameter Sweep

The `sweep_hyperparameters.py` script allows you to run hyperparameter sweeps over gamma and rank values for a given dataset.

### Checkpoint-Based Mode

If you have a Stage 1 checkpoint, you can sweep directly:

```bash
python scripts/sweep_hyperparameters.py \
    --dataset waterbirds \
    --data_dir /path/to/waterbirds \
    --checkpoint /path/to/stage1_checkpoint.pt \
    --model imagenet_resnet50_pretrained \
    --num_classes 2 \
    --gamma_list "5.0,10.0,15.0,20.0" \
    --rank_list "1,5,10,20" \
    --sweep_output_dir ./logs/marla_sweep \
    --num_epochs 1000 \
    --lr 0.01 \
    --seed 42
```

### Embedding-Based Mode

If you have pre-extracted embeddings in a directory:

```bash
python scripts/sweep_hyperparameters.py \
    --dataset waterbirds \
    --data_dir /path/to/waterbirds \
    --embedding_dir /path/to/embeddings \
    --gamma_list "5.0,10.0,15.0,20.0" \
    --rank_list "1,5,10,20" \
    --sweep_output_dir ./logs/marla_sweep \
    --num_epochs 1000 \
    --lr 0.01 \
    --seed 42
```

The embedding directory should contain:
- `train_embeddings.pt`, `train_labels.pt` (and optionally `train_groups.pt`)
- `val_embeddings.pt`, `val_labels.pt` (and optionally `val_groups.pt`)
- `test_embeddings.pt`, `test_labels.pt` (and optionally `test_groups.pt`)
- `base_weight.pt`, `base_bias.pt` (optional)

### Sweep Results

Results are organized in the sweep output directory:
```
logs/marla_sweep/
└── waterbirds/
    ├── sweep_config.json          # Sweep configuration
    ├── sweep_results.json          # Summary of all experiments
    ├── gamma5.0_rank1/            # Individual experiment results
    ├── gamma5.0_rank5/
    ├── gamma10.0_rank1/
    └── ...
```

Each experiment directory contains:
- Training logs and metrics
- Checkpoints (if saved)
- `results.json` with final metrics

The script automatically:
- Runs all combinations of gamma and rank values
- Organizes results in separate directories
- Collects and summarizes results
- Identifies the best configuration based on worst-group accuracy

### Additional Options

- `--use_wandb`: Enable Weights & Biases logging for all experiments
- `--wandb_project`: W&B project name (default: `marla_sweep`)
- `--parallel`: Run experiments in parallel (requires job management)
- `--max_parallel`: Maximum parallel jobs (default: 4)

## Dataset Setup

MARLA automatically generates `metadata.csv` files from raw datasets. Simply point to your dataset directory and metadata will be created automatically.

### CelebA
**Required files:**
- `list_eval_partition.txt` (train/val/test splits)
- `list_attr_celeba.txt` (attributes)
- `img_align_celeba/` directory with images

**Usage:**
```bash
python -m marla.train_stage1 --dataset celeba --data_dir /path/to/celeba
```

### Waterbirds
**Required files:**
- `waterbird_complete95_forest2water2/metadata.csv` (original metadata)

**Usage:**
```bash
python -m marla.train_stage1 --dataset waterbirds --data_dir /path/to/waterbirds
```

### MultiNLI
**Required files:**
- `data/metadata_random.csv` (or `metadata_random.csv` in root)

**Usage:**
```bash
python -m marla.train_stage1 --dataset multinli --data_dir /path/to/multinli
```

### CivilComments
**Required files:**
- `all_data_with_identities.csv` (WILDS format)

**Usage:**
```bash
python -m marla.train_stage1 --dataset civilcomments --data_dir /path/to/civilcomments
```

### Manual Metadata Generation

If you want to generate metadata manually (e.g., before training):

```bash
python -m marla.data.generate_metadata --dataset celeba --data_dir /path/to/celeba
python -m marla.data.generate_metadata --dataset waterbirds --data_dir /path/to/waterbirds
python -m marla.data.generate_metadata --dataset multinli --data_dir /path/to/multinli
python -m marla.data.generate_metadata --dataset civilcomments --data_dir /path/to/civilcomments --granularity coarse
```

The generated `metadata.csv` files will have columns: `id, filename, split, y, a` where:
- `id`: Unique sample ID
- `filename`: Path to data file (relative to dataset directory)
- `split`: 0=train, 1=val, 2=test
- `y`: Class label
- `a`: Attribute/group label

## Key Parameters

- `--spectral_rank` (k): Dimension of error subspace (default: 1) (typically tuned with hyperparameter sweeps)
- `--gamma`: Reweighting strength for error-informed weights (default: 10.0) (typically tuned with hyperparameter sweeps)
- `--reg_coeff`: Regularization coefficient for adjustment matrix (default: 0.0)
- `--lr`: Learning rate for adjustment matrix (default: 0.01)
- `--num_epochs`: Number of training epochs (default: 1000 for Stage 2)

## How MARLA Works

1. **Error-Informed Weights**: Computes weights `μ_i = exp(γ * (1 - p_i))` where `p_i` is the predicted probability of the correct label. Higher weight is assigned to examples with lower confidence.

2. **Error-Weighted Covariance**: Computes `Σ_err = Σ_i μ_i (φ(x_i) - φ̄)(φ(x_i) - φ̄)^T` where `φ(x_i)` are embeddings and `φ̄` is the weighted mean.

3. **Spectral Decomposition**: Extracts top-k eigenvectors `V_k` from `Σ_err`. These eigenvectors span the k-dimensional error subspace.

4. **Low-Rank Adjustment**: Learns adjustment matrix `A ∈ ℝ^(k×C)` such that:
   - `base_logits = W φ(x) + b` (frozen base classifier)
   - `projected = V_k^T φ(x)` (project to error subspace)
   - `adjustment_logits = A^T projected` (learned adjustment)
   - `adjusted_logits = base_logits + adjustment_logits` (final prediction)
