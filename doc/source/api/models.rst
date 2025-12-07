======
models
======

Model Factories
===============

.. automodule:: mylib.models
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The :mod:`mylib.models` module provides factory functions for creating neural network models 
suitable for feature selection experiments.

create_classification_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mylib.models.create_classification_model

Creates a feedforward neural network classifier with the following architecture:

**Architecture:**

.. code-block:: text

   Input (n_features)
       │
   Linear (n_features -> hidden_dim)
       │
   BatchNorm1d
       │
   ReLU
       │
   Dropout(0.3)
       │
   Linear (hidden_dim -> hidden_dim//2)
       │
   BatchNorm1d
       │
   ReLU
       │
   Dropout(0.2)
       │
   Linear (hidden_dim//2 -> num_classes)
       │
   Output Logits (num_classes)

**Design Decisions:**

1. **Auto-calculated hidden dimensions**
   - Prevents manually specifying size for different problems
   - Hidden size: :math:`\min(128, \max(64, n_{features}))`
   - Balances capacity and overfitting risk

2. **BatchNorm layers**
   - Stabilizes training
   - Allows higher learning rates
   - Reduces internal covariate shift

3. **Dropout for regularization**
   - Reduces overfitting
   - Rates decrease in deeper layers (0.3 → 0.2)
   - Standard practice for neural networks

4. **Two hidden layers**
   - Sufficient capacity for most problems
   - Avoids extreme depth (slow training)
   - Good for feature selection experiments

**Example Usage:**

.. code-block:: python

   from mylib import create_classification_model
   
   # Binary classification with 100 features
   model = create_classification_model(
       input_dim=100,
       num_classes=2
   )
   
   # Multi-class (10 classes) with custom hidden dimension
   model = create_classification_model(
       input_dim=784,      # e.g., MNIST flattened
       num_classes=10,
       hidden_dim=256
   )

**Output:**

The model returns logits (unnormalized scores) that should be passed through softmax 
or used with CrossEntropyLoss (which applies softmax internally).

.. code-block:: python

   import torch
   
   x = torch.randn(32, 100)  # batch_size=32, features=100
   logits = model(x)         # shape: [32, 2]
   
   # Use with CrossEntropyLoss
   loss = nn.CrossEntropyLoss()(logits, y)

Custom Models
~~~~~~~~~~~~~

For custom architectures, you can implement your own model:

.. code-block:: python

   import torch.nn as nn
   
   class CustomModel(nn.Module):
       def __init__(self, input_dim: int, num_classes: int):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(input_dim, 256),
               nn.ReLU(),
               nn.Linear(256, 128),
               nn.ReLU(),
               nn.Linear(128, num_classes)
           )
       
       def forward(self, x):
           return self.layers(x)
   
   model = CustomModel(input_dim=100, num_classes=2)
   trainer = FeatureSelectionTrainer(model=model, selector=selector, ...)
