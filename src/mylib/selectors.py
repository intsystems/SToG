"""Feature selector implementations."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .base import BaseFeatureSelector


class STGLayer(BaseFeatureSelector):
    """
    Stochastic Gates (STG) - Original implementation from Yamada et al. 2020.
    Uses Gaussian-based continuous relaxation of Bernoulli variables.
    
    Reference: "Learning Feature Sparse Principal Subspace" (Yamada et al., ICML 2020)
    """
    
    def __init__(self, input_dim: int, sigma: float = 0.5, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.mu = nn.Parameter(torch.zeros(input_dim))
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic gates to input features."""
        if self.training:
            eps = torch.randn_like(self.mu) * self.sigma
            z = self.mu + eps
            gates = torch.clamp(z, 0.0, 1.0)
        else:
            # At inference, use deterministic mean
            gates = torch.clamp(self.mu, 0.0, 1.0)
        return x * gates.unsqueeze(0)

    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization: sum of probabilities of selection."""
        # P(z_d > 0) = 0.5 * (1 + erf(mu_d / (sigma * sqrt(2))))
        return torch.sum(0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2)))))

    def get_selection_probs(self) -> torch.Tensor:
        """Get selection probabilities for each feature."""
        return (0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2))))).detach()


class STELayer(BaseFeatureSelector):
    """
    Straight-Through Estimator for feature selection.
    Uses binary gates with gradient flow through sigmoid.
    
    Reference: "Estimating or Propagating Gradients Through Stochastic 
    Neurons for Conditional Computation" (Bengio et al., 2013)
    """
    
    def __init__(self, input_dim: int, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.logits = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply straight-through gates to input features."""
        probs = torch.sigmoid(self.logits)
        
        if self.training:
            # Straight-through: hard binary in forward, soft gradient in backward
            gates_hard = (probs > 0.5).float()
            gates = gates_hard - probs.detach() + probs
        else:
            gates = (probs > 0.5).float()
        
        return x * gates.unsqueeze(0)

    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization: sum of selection probabilities."""
        return torch.sum(torch.sigmoid(self.logits))

    def get_selection_probs(self) -> torch.Tensor:
        """Get selection probabilities for each feature."""
        return torch.sigmoid(self.logits).detach()


class GumbelLayer(BaseFeatureSelector):
    """
    Gumbel-Softmax based feature selector.
    Uses categorical distribution over {off, on} for each feature.
    
    Reference: "Categorical Reparameterization with Gumbel-Softmax"
    (Jang et al., ICLR 2017)
    
    Fixed implementation: Properly handles batch dimension and sampling.
    """
    
    def __init__(self, input_dim: int, temperature: float = 1.0, device: str = 'cpu'):
        super().__init__(input_dim, device)
        # Initialize with bias toward "off" state (first column larger)
        # This encourages sparsity initially
        self.logits = nn.Parameter(torch.zeros(input_dim, 2))
        self.logits.data[:, 0] = 1.0  # Bias toward off state
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gumbel-Softmax gates to input features.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Gated input tensor
        """
        if self.training:
            sampled = F.gumbel_softmax(
                self.logits, 
                tau=self.temperature, 
                hard=True, 
                dim=1
            )
            gates = sampled[:, 1]
        else:
            gates = (self.logits[:, 1] > self.logits[:, 0]).float()
        return x * gates.unsqueeze(0)

    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization: sum of "on" state probabilities."""
        probs = F.softmax(self.logits, dim=1)[:, 1]
        return torch.sum(probs)

    def get_selection_probs(self) -> torch.Tensor:
        """Get selection probabilities for each feature."""
        return F.softmax(self.logits, dim=1)[:, 1].detach()
    
    def set_temperature(self, temperature: float):
        """Update temperature for annealing schedule."""
        self.temperature = temperature


class CorrelatedSTGLayer(BaseFeatureSelector):
    """
    STG with explicit handling of correlated features.
    Based on "Adaptive Group Sparse Regularization for Deep Neural Networks".
    Uses group structure to handle feature correlation.
    
    Reference: "Adaptive Group Sparse Regularization for Deep Neural Networks"
    """
    
    def __init__(self, input_dim: int, sigma: float = 0.5, 
                 group_penalty: float = 0.1, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.mu = nn.Parameter(torch.zeros(input_dim))
        self.sigma = sigma
        self.group_penalty = group_penalty
        
        self.correlation_weights = nn.Parameter(torch.eye(input_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply correlated stochastic gates to input features."""
        if self.training:
            eps = torch.randn_like(self.mu) * self.sigma
            z = self.mu + eps
            gates = torch.clamp(z, 0.0, 1.0)
        else:
            gates = torch.clamp(self.mu, 0.0, 1.0)
        
        return x * gates.unsqueeze(0)

    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization with correlation penalty."""
        base_reg = torch.sum(0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2)))))
        
        probs = 0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2))))
        
        prob_diff = probs.unsqueeze(0) - probs.unsqueeze(1)
        correlation_penalty = torch.sum(torch.abs(self.correlation_weights) * prob_diff ** 2)
        
        return base_reg + self.group_penalty * correlation_penalty

    def get_selection_probs(self) -> torch.Tensor:
        """Get selection probabilities for each feature."""
        return (0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2))))).detach()


class L1Layer(BaseFeatureSelector):
    """
    L1 regularization on input layer weights.
    Baseline comparison method for feature selection.
    """
    
    def __init__(self, input_dim: int, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.weights = nn.Parameter(torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply L1 weights to input features."""
        return x * self.weights.unsqueeze(0)

    def regularization_loss(self) -> torch.Tensor:
        """Compute L1 regularization: sum of absolute weights."""
        return torch.sum(torch.abs(self.weights))

    def get_selection_probs(self) -> torch.Tensor:
        """Get feature importance (absolute weights)."""
        return torch.abs(self.weights).detach()

    def get_selected_features(self, threshold: float = 0.1) -> np.ndarray:
        """Get selected features based on weight magnitude."""
        probs = self.get_selection_probs()
        return (probs > threshold).cpu().numpy()

