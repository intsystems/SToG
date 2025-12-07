=======
trainer
=======

Training Utilities
==================

.. automodule:: mylib.trainer
   :members:
   :undoc-members:
   :show-inheritance:

FeatureSelectionTrainer
=======================

.. autoclass:: mylib.trainer.FeatureSelectionTrainer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

The :class:`mylib.trainer.FeatureSelectionTrainer` handles joint optimization of a classification 
model and a feature selector. It implements:

- **Two-optimizer approach** - Separate optimizers for model and selector
- **Early stopping** - Validation-based stopping with configurable patience
- **Gradient clipping** - Prevents gradient explosion
- **History tracking** - Records metrics for analysis
- **Model checkpointing** - Saves best model state

Architecture
~~~~~~~~~~~~

.. code-block:: text

   Input Data
       │
       ├─> Selector (Feature Gates)
       │        │
       │   [Gate Parameters]
       │
       └─> Model (Classifier)
               │
           [Model Parameters]
               │
          Output Logits
               │
          Classification Loss + Regularization Loss
               │
          ┌─────┴──────┐
          │             │
       Model        Selector
      Optimizer     Optimizer
      (lr=0.001)    (lr=0.01)
          │             │
          └─────┬───────┘
                │
         Update Parameters

Joint Loss Function
~~~~~~~~~~~~~~~~~~~

The trainer optimizes:

.. math::

   \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(\mathbf{f}, \mathbf{g}) + \lambda \Omega(\mathbf{g})

where:

- :math:`\mathcal{L}_{\text{task}}` is the classification loss (CrossEntropyLoss)
- :math:`\Omega(\mathbf{g})` is the regularization from selector
- :math:`\lambda` controls sparsity-accuracy trade-off

Two-Optimizer Strategy
~~~~~~~~~~~~~~~~~~~~~~

**Model Optimizer:**
   - Lower learning rate (default: 0.001)
   - Updates classification parameters :math:`\mathbf{f}`
   - Learns from task loss

**Selector Optimizer:**
   - Higher learning rate (default: 0.01)
   - Updates gate parameters :math:`\mathbf{g}`
   - Learns from task + regularization loss
   - 10x higher learning rate enables faster adaptation

Early Stopping
~~~~~~~~~~~~~~

Early stopping monitors
