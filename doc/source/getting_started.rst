Getting Started
===============

Overview
--------

This library provides a complete implementation of stochastic gating methods for feature selection in neural networks. It includes all major stochastic gating approaches:

1. **STG (Stochastic Gates)** - Original Gaussian-based method from Yamada et al. 2020
2. **STE (Straight-Through Estimator)** - Binary gates with gradient approximation  
3. **Gumbel-Softmax** - Categorical relaxation for feature gating
4. **Correlated STG** - Extension for handling correlated features
5. **L1 Regularization** - Baseline comparison method

Installation
------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.7 or higher
- pip package manager

Quick Setup
~~~~~~~~~~~

1. **Create virtual environment:**
   .. code-block:: bash
   
      python -m venv venv

2. **Activate virtual environment:**
   
   Windows:
   .. code-block:: batch
   
      venv\Scripts\activate
   
   Linux/Mac:
   .. code-block:: bash
   
      source venv/bin/activate

3. **Install dependencies:**
   .. code-block:: bash
   
      pip install -r requirements.txt
   
   Or using uv (faster):
   .. code-block:: bash
   
      pip install uv
      uv pip install -r requirements.txt

4. **Install package in editable mode:**
   .. code-block:: bash
   
      pip install -e .

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from mylib import (
       STGLayer, 
       FeatureSelectionTrainer, 
       create_classification_model
   )
   import torch.nn as nn
   
   # Create model and selector
   n_features = 100
   n_classes = 2
   model = create_classification_model(n_features, n_classes)
   selector = STGLayer(n_features, sigma=0.5)
   
   # Train
   trainer = FeatureSelectionTrainer(
       model=model,
       selector=selector,
       criterion=nn.CrossEntropyLoss(),
       lambda_reg=0.05
   )
   
   trainer.fit(X_train, y_train, X_val, y_val, epochs=300)
   
   # Evaluate
   result = trainer.evaluate(X_test, y_test)
   print(f"Accuracy: {result['test_acc']:.2f}%")
   print(f"Selected: {result['selected_count']} features")
   print(f"Selected features: {result['selected_features']}")

Using Different Selectors
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mylib import STGLayer, STELayer, GumbelLayer, CorrelatedSTGLayer, L1Layer
   
   # STG with custom sigma
   selector = STGLayer(input_dim=100, sigma=0.5)
   
   # Straight-Through Estimator
   selector = STELayer(input_dim=100)
   
   # Gumbel-Softmax with temperature
   selector = GumbelLayer(input_dim=100, temperature=1.0)
   # Can update temperature during training
   selector.set_temperature(0.5)
   
   # Correlated STG
   selector = CorrelatedSTGLayer(input_dim=100, sigma=0.5, group_penalty=0.1)
   
   # L1 baseline
   selector = L1Layer(input_dim=100)

Load Datasets
~~~~~~~~~~~~~

.. code-block:: python

   from mylib import DatasetLoader
   
   loader = DatasetLoader()
   
   # Load built-in datasets
   breast_cancer = loader.load_breast_cancer()
   wine = loader.load_wine()
   synthetic_high_dim = loader.create_synthetic_high_dim()
   synthetic_correlated = loader.create_synthetic_correlated()
   
   # Access data
   X = breast_cancer['X']
   y = breast_cancer['y']
   print(f"Dataset: {breast_cancer['name']}")
   print(f"Shape: {X.shape}")
   print(f"Description: {breast_cancer['description']}")

Run Full Benchmark
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mylib import ComprehensiveBenchmark, DatasetLoader
   
   # Load datasets
   loader = DatasetLoader()
   datasets = [
       loader.load_breast_cancer(),
       loader.load_wine(),
       loader.create_synthetic_high_dim(),
       loader.create_synthetic_correlated()
   ]
   
   # Run benchmark
   benchmark = ComprehensiveBenchmark(device='cpu')
   benchmark.run_benchmark(datasets)
   
   # Or use default datasets
   benchmark.run_benchmark()  # Uses all built-in datasets

