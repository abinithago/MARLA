"""
Model architectures for MARLA datasets.

Provides model factories for:
- Image models: ResNet50 (for CelebA, Waterbirds)
- Text models: BERT (for CivilComments, MultiNLI)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertForSequenceClassification, BertConfig


def imagenet_resnet50_pretrained(output_dim):
    """
    Create ResNet50 pretrained on ImageNet.
    
    Args:
        output_dim: Number of output classes
    
    Returns:
        ResNet50 model with final layer replaced for output_dim classes
    """
    model = models.resnet50(pretrained=True)
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, output_dim)
    return model


def imagenet_resnet50(output_dim):
    """
    Create ResNet50 without pretraining.
    
    Args:
        output_dim: Number of output classes
    
    Returns:
        ResNet50 model with final layer replaced for output_dim classes
    """
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, output_dim)
    return model


def bert_base_uncased(output_dim, num_labels=None):
    """
    Create BERT-base-uncased model for sequence classification.
    
    Args:
        output_dim: Number of output classes (deprecated, use num_labels)
        num_labels: Number of output classes (default: output_dim or 2)
    
    Returns:
        BERT model for sequence classification
    """
    if num_labels is None:
        num_labels = output_dim if output_dim else 2
    
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels
    )
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        config=config
    )
    return model


def get_model(model_name, output_dim, **kwargs):
    """
    Factory function to get model by name.
    
    Args:
        model_name: Name of the model
        output_dim: Number of output classes
        **kwargs: Additional arguments for specific models
    
    Returns:
        Model instance
    """
    model_map = {
        'imagenet_resnet50_pretrained': imagenet_resnet50_pretrained,
        'imagenet_resnet50': imagenet_resnet50,
        'bert-base-uncased': bert_base_uncased,
        'bert': bert_base_uncased,
    }
    
    if model_name not in model_map:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(model_map.keys())}"
        )
    
    model_fn = model_map[model_name]
    return model_fn(output_dim, **kwargs)


def get_classifier_and_feature_extractor(model):
    """
    Split model into feature extractor and classifier.
    
    For ResNet: feature_extractor = model without fc, classifier = fc
    For BERT: feature_extractor = model without classifier, classifier = classifier
    
    Args:
        model: Full model
    
    Returns:
        feature_extractor: Model that outputs embeddings
        classifier: Linear layer that outputs logits
    """
    # Check if it's a ResNet (image model)
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        # ResNet architecture
        classifier = model.fc
        # Create feature extractor without final layer
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        # Add flattening
        feature_extractor = nn.Sequential(
            feature_extractor,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        return feature_extractor, classifier
    
    # Check if it's a BERT model (text model)
    elif hasattr(model, 'classifier') and hasattr(model, 'bert'):
        # BERT architecture
        classifier = model.classifier
        # Create feature extractor (BERT without classifier)
        class BERTFeatureExtractor(nn.Module):
            def __init__(self, bert_model):
                super().__init__()
                self.bert = bert_model.bert
                self.config = bert_model.config
            
            def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    **kwargs
                )
                # Return pooled output (CLS token representation)
                # If pooler_output is None, use mean pooling over sequence
                if outputs.pooler_output is not None:
                    return outputs.pooler_output
                else:
                    # Mean pooling over sequence length
                    last_hidden_state = outputs.last_hidden_state
                    if attention_mask is not None:
                        mask = attention_mask.unsqueeze(-1).float()
                        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                    else:
                        pooled = last_hidden_state.mean(dim=1)
                    return pooled
        
        feature_extractor = BERTFeatureExtractor(model)
        return feature_extractor, classifier
    
    else:
        raise ValueError(
            f"Unknown model architecture. "
            f"Expected ResNet (with 'fc') or BERT (with 'classifier' and 'bert'), "
            f"but got: {type(model)}"
        )
