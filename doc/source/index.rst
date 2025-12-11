==============================================
SToG: Stochastic Gating for Feature Selection
==============================================

**Feature selection using stochastic gating methods for neural networks**

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   overview
   installation
   quick_start
   tutorial
   api/index
   references

Welcome to SToG
===============

SToG is a PyTorch library implementing stochastic gating methods for automatic feature selection in neural networks. 
The library provides implementations of several state-of-the-art feature selection techniques, from classical 
L1 regularization to modern continuous relaxation methods.

Key Features
~~~~~~~~~~~~

- **Multiple stochastic gating methods** - STG, STE, Gumbel-Softmax, and variants
- **Correlated feature handling** - Explicit support for redundant and correlated features  
- **Flexible architecture** - Base class for implementing custom selectors
- **Comprehensive benchmarking** - Built-in benchmarking framework for comparison
- **PyTorch native** - Full integration with PyTorch ecosystem

Core Methods Implemented
~~~~~~~~~~~~~~~~~~~~~~~~

- **STG (Stochastic Gates)** - Gaussian-based continuous relaxation for binary gates
- **STE (Straight-Through Estimator)** - Binary gates with gradient flow approximation
- **Gumbel-Softmax** - Categorical distribution relaxation with temperature annealing
- **CorrelatedSTG** - STG variant for redundant feature groups
- **L1 Regularization** - Baseline method for comparison

Quick Navigation
================

.. code-block:: python

   import torch
   from SToG import STGLayer, FeatureSelectionTrainer, create_classification_model

   # Create model and selector
   model = create_classification_model(n_features=100, n_classes=2)
   selector = STGLayer(n_features=100, sigma=0.5)

   # Train with feature selection
   trainer = FeatureSelectionTrainer(
       model=model,
       selector=selector,
       criterion=torch.nn.CrossEntropyLoss(),
       lambda_reg=0.05
   )
   
   trainer.fit(X_train, y_train, X_val, y_val, epochs=300)
   result = trainer.evaluate(X_test, y_test)
   print(f"Accuracy: {result['test_acc']:.1f}%")
   print(f"Selected features: {result['selected_count']}")

Next Steps
~~~~~~~~~~

- :doc:`overview` - Problem formulation and methodology overview
- :doc:`installation` - Installation and setup instructions
- :doc:`quick_start` - Quick start guide with basic examples
- :doc:`tutorial` - Detailed tutorial with real examples
- :doc:`api/index` - Complete API reference
- :doc:`references` - Academic references and citations

.. toctree::
   :maxdepth: 1
   :hidden:

   self
