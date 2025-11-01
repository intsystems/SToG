"""Base classes for feature selection methods."""
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


class BaseFeatureSelector(nn.Module, ABC):
    """Base class for feature selection methods."""
    
    def __init__(self, input_dim: int, device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature selection gates to input.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Gated input tensor of shape [batch_size, input_dim]
        """
        pass

    @abstractmethod
    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss for sparsity.
        
        Returns:
            Scalar tensor with regularization loss
        """
        pass

    @abstractmethod
    def get_selection_probs(self) -> torch.Tensor:
        """Get feature selection probabilities.
        
        Returns:
            Tensor of shape [input_dim] with selection probabilities
        """
        pass

    def get_selected_features(self, threshold: float = 0.5) -> np.ndarray:
        """Get binary mask of selected features.
        
        Args:
            threshold: Probability threshold for selection
            
        Returns:
            Boolean array of shape [input_dim]
        """
        probs = self.get_selection_probs()
        return (probs > threshold).cpu().numpy()

