==============================================
SToG: Stochastic Gating for Feature Selection
==============================================

**Feature selection using stochastic gating methods for neural networks**

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   installation
   train
   api/index
   info

Welcome
=======

SToG is a PyTorch library implementing stochastic gating methods for feature selection.

Key Methods
-----------

- **STG** (Stochastic Gates) - Gaussian-based continuous relaxation
- **STE** (Straight-Through Estimator) - Binary gates with gradient flow
- **Gumbel-Softmax** - Categorical distribution relaxation
- **Correlated STG** - For redundant/correlated features
- **L1** - Baseline L1 regularization

Quick Start
===========

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install stog

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from mylib import STGLayer, FeatureSelectionTrainer, create_classification_model

   # Create model and selector
   model = create_classification_model(n_features=100, n_classes=2)
   selector = STGLayer(n_features=100, sigma=0.5)

   # Train
   trainer = FeatureSelectionTrainer(
       model=model,
       selector=selector,
       criterion=torch.nn.CrossEntropyLoss(),
       lambda_reg=0.05
   )
   
   trainer.fit(X_train, y_train, X_val, y_val, epochs=300)
   result = trainer.evaluate(X_test, y_test)

Next Steps
==========

- :doc:`installation` - Installation guide
- :doc:`train` - Training and benchmarking
- :doc:`api/index` - API Reference
- :doc:`info` - About and citation

.. toctree::
   :maxdepth: 1
   :hidden:

   self
