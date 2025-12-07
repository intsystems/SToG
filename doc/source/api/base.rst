====
base
====

Base Feature Selector Class
============================

.. automodule:: mylib.base
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The :class:`mylib.base.BaseFeatureSelector` is an abstract base class that defines the interface 
for all feature selection methods in SToG. All concrete selector implementations must inherit 
from this class and implement the required abstract methods.

Key Methods
~~~~~~~~~~~

**forward(x)** - Apply feature gating
   Applies learned gate parameters to input features, returning gated input.

**regularization_loss()** - Compute sparsity regularization
   Returns a scalar loss that encourages sparse feature selection.

**get_selection_probs()** - Get selection probabilities
   Returns per-feature selection probabilities used for determining which features are important.

**get_selected_features(threshold)** - Get binary selection mask
   Returns a binary mask indicating selected vs. discarded features.

Design Pattern
~~~~~~~~~~~~~~

All selectors follow this pattern:

1. **Initialization** - Set up learnable parameters
2. **Forward pass** - Apply gates to input during training/inference
3. **Regularization** - Compute sparsity-inducing loss
4. **Interpretation** - Extract feature importance from learned parameters

Example Implementation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mylib.base import BaseFeatureSelector
   import torch
   import torch.nn as nn
   
   class CustomSelector(BaseFeatureSelector):
       """Custom feature selector implementation."""
       
       def __init__(self, input_dim: int, device: str = 'cpu'):
           super().__init__(input_dim, device)
           self.weights = nn.Parameter(torch.randn(input_dim))
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Gate features using learned weights."""
           gates = torch.sigmoid(self.weights)
           return x * gates.unsqueeze(0)
       
       def regularization_loss(self) -> torch.Tensor:
           """Sparsity regularization."""
           return torch.sum(torch.sigmoid(self.weights))
       
       def get_selection_probs(self) -> torch.Tensor:
           """Selection probabilities."""
           return torch.sigmoid(self.weights).detach()
