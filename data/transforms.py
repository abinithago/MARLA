"""
Data augmentation transforms for MARLA datasets.

Provides transforms for:
- CelebA and Waterbirds (image datasets)
- CivilComments and MultiNLI (text datasets)
"""

import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertTokenizerFast, DistilBertTokenizerFast

# ImageNet normalization statistics (standard for pretrained models)
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _add_totensor_normalize(transform_lst, normalize_stats):
    """Helper to add ToTensor and Normalize to transform list."""
    transform_lst.append(transforms.ToTensor())
    if normalize_stats:
        transform_lst.append(transforms.Normalize(*normalize_stats))


class BaseWaterbirdsCelebATransform(transforms.Compose):
    """
    Base transform for Waterbirds and CelebA datasets.
    
    For training: RandomResizedCrop + RandomHorizontalFlip
    For validation/test: Resize + CenterCrop
    """
    def __init__(self, augment, normalize_stats):
        target_resolution = (224, 224)
        resize_resolution = (256, 256)
        self.transforms = []
        if augment:
            self.transforms = [
                transforms.RandomResizedCrop(
                    target_resolution, 
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333), 
                    interpolation=2  # BILINEAR (matching AFR)
                ),
                transforms.RandomHorizontalFlip()
            ]
        else:
            self.transforms = [
                transforms.Resize(resize_resolution),
                transforms.CenterCrop(target_resolution)
            ]
        _add_totensor_normalize(self.transforms, normalize_stats)


class AugWaterbirdsCelebATransform(BaseWaterbirdsCelebATransform):
    """
    Augmented transform for Waterbirds/CelebA (with data augmentation).
    
    Used during Stage 1 training (supervised training).
    """
    def __init__(self, train=True):
        super().__init__(augment=train, normalize_stats=IMAGENET_STATS)


class NoAugWaterbirdsCelebATransform(BaseWaterbirdsCelebATransform):
    """
    No-augmentation transform for Waterbirds/CelebA.
    
    Used during Stage 2 (embedding extraction and MARLA training).
    """
    def __init__(self, train=True):
        super().__init__(augment=False, normalize_stats=IMAGENET_STATS)


class NoAugNoNormWaterbirdsCelebATransform(BaseWaterbirdsCelebATransform):
    """
    No-augmentation, no-normalization transform for Waterbirds/CelebA.
    
    Used when normalization is not desired.
    """
    def __init__(self, train=True):
        super().__init__(augment=False, normalize_stats=None)


class ImageNetAugmixTransform(transforms.Compose):
    """
    AugMix transform for ImageNet-style datasets.
    
    Uses AugMix augmentation strategy for improved robustness.
    """
    def __init__(self, train=True):
        # Import here to avoid circular dependency
        from .augmix import AugmixTransformBase
        
        preprocess = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(*IMAGENET_STATS)
        ])
        
        if train:
            self.transforms = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                AugmixTransformBase(preprocess=preprocess)
            ]
        else:
            self.transforms = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                preprocess
            ]


class BertTokenizeTransform:
    """
    BERT tokenization transform for text datasets (CivilComments, MultiNLI).
    
    Tokenizes text using BERT tokenizer and returns input_ids, attention_mask,
    and token_type_ids stacked together.
    """
    def __init__(self, train=True, model_name="bert-base-uncased", max_length=220):
        """
        Args:
            train: Whether this is for training (unused, kept for compatibility)
            model_name: BERT model name (default: "bert-base-uncased")
            max_length: Maximum sequence length (default: 220)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def __call__(self, text):
        """Tokenize text and return stacked tensors."""
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return torch.squeeze(
            torch.stack((
                tokens["input_ids"], 
                tokens["attention_mask"], 
                tokens["token_type_ids"]
            ), dim=2), 
            dim=0
        )


class BertTransform:
    """
    Picklable BERT transform class for multiprocessing.
    
    This class can be pickled and sent to worker processes.
    Lazy-loads tokenizer to avoid issues with multiprocessing.
    """
    def __init__(self, model_name="bert-base-uncased", max_token_length=220):
        self.model_name = model_name
        self.max_token_length = max_token_length
        self._tokenizer = None
    
    def _get_tokenizer(self):
        """Lazy load tokenizer (needed for multiprocessing with spawn)."""
        if self._tokenizer is None:
            if self.model_name == "bert-base-uncased":
                self._tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
            elif self.model_name == "distilbert-base-uncased":
                self._tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
            else:
                raise ValueError(f"Model: {self.model_name} not recognized.")
        return self._tokenizer
    
    def __call__(self, text):
        """Transform text to tokenized tensor."""
        tokenizer = self._get_tokenizer()
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt",
        )
        if self.model_name == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif self.model_name == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x


def get_transform(transform_name, train=True, **kwargs):
    """
    Factory function to get transform by name.
    
    Args:
        transform_name: Name of the transform
        train: Whether this is for training
        **kwargs: Additional arguments for specific transforms
    
    Returns:
        Transform object
    """
    transform_map = {
        'AugWaterbirdsCelebATransform': AugWaterbirdsCelebATransform,
        'NoAugWaterbirdsCelebATransform': NoAugWaterbirdsCelebATransform,
        'NoAugNoNormWaterbirdsCelebATransform': NoAugNoNormWaterbirdsCelebATransform,
        'ImageNetAugmixTransform': ImageNetAugmixTransform,
        'BertTokenizeTransform': BertTokenizeTransform,
        'BertTransform': BertTransform,
    }
    
    if transform_name not in transform_map:
        raise ValueError(
            f"Unknown transform: {transform_name}. "
            f"Available: {list(transform_map.keys())}"
        )
    
    transform_cls = transform_map[transform_name]
    
    # Handle different transform signatures
    if transform_name in ['BertTokenizeTransform', 'BertTransform']:
        return transform_cls(train=train, **kwargs)
    else:
        return transform_cls(train=train)
