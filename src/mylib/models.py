"""Model definitions for classification tasks."""
import torch
import torch.nn as nn


def create_classification_model(input_dim: int, num_classes: int, 
                                hidden_dim: int = None) -> nn.Module:
    """
    Create a simple feedforward neural network for classification.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension (auto-calculated if None)
        
    Returns:
        PyTorch sequential model
    """
    if hidden_dim is None:
        hidden_dim = min(128, max(64, input_dim))
    
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.BatchNorm1d(hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim // 2, num_classes)
    )

