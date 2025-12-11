=============
API Reference
=============

Complete API documentation for SToG.

.. toctree::
   :maxdepth: 2

   base
   selectors
   trainer
   models
   datasets
   benchmark
   main

Module Overview
===============

The SToG library consists of several interconnected modules:

**base.py** - Abstract base class
   Defines :class:`SToG.base.BaseFeatureSelector`, the abstract base for all feature selector implementations.

**selectors.py** - Feature selection methods
   Implements five feature selection methods:
   
   - :class:`SToG.selectors.STGLayer` - Stochastic Gates with Gaussian relaxation
   - :class:`SToG.selectors.STELayer` - Straight-Through Estimator
   - :class:`SToG.selectors.GumbelLayer` - Gumbel-Softmax categorical relaxation
   - :class:`SToG.selectors.CorrelatedSTGLayer` - STG for correlated features
   - :class:`SToG.selectors.L1Layer` - L1 regularization baseline

**trainer.py** - Training utilities
   Provides :class:`SToG.trainer.FeatureSelectionTrainer` for joint optimization of model and selector.

**models.py** - Model factories
   Provides :func:`SToG.models.create_classification_model` for creating neural network classifiers.

**datasets.py** - Dataset utilities
   Provides :class:`SToG.datasets.DatasetLoader` for loading and preparing datasets.

**benchmark.py** - Benchmarking framework
   Provides :class:`SToG.benchmark.ComprehensiveBenchmark` for comparing methods across datasets.

**main.py** - Main execution
   Entry point for running benchmarks via :func:`SToG.main.main`.

Design Philosophy
=================

**Modular Architecture**

Each feature selector inherits from :class:`SToG.base.BaseFeatureSelector`, ensuring consistent interface:

.. code-block:: text

   BaseFeatureSelector (Abstract)
   ├── forward(x) -> x_gated
   ├── regularization_loss() -> scalar
   ├── get_selection_probs() -> probabilities
   └── get_selected_features(threshold) -> mask

**Extensibility**

New feature selection methods can be implemented by subclassing :class:`SToG.base.BaseFeatureSelector` 
and implementing three methods: ``forward``, ``regularization_loss``, and ``get_selection_probs``.

**PyTorch Integration**

All components are built on PyTorch:

- Selectors inherit from :class:`torch.nn.Module`
- Computations use standard PyTorch tensors
- Compatible with PyTorch's optimization and autograd system
