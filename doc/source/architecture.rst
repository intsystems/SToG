Architecture
============

Project Structure
-----------------

::

   SToG/
   ├── src/
   │   └── mylib/              # Main library package
   │       ├── __init__.py     # Package exports
   │       ├── base.py         # BaseFeatureSelector abstract class
   │       ├── selectors.py    # All feature selector implementations
   │       ├── trainer.py      # FeatureSelectionTrainer class
   │       ├── models.py       # Model creation utilities
   │       ├── datasets.py     # Dataset loading utilities
   │       ├── benchmark.py    # Benchmarking framework
   │       └── main.py         # Main execution script
   ├── test/
   │   └── test_stochastic_gating.py  # Test suite
   ├── requirements.txt        # Python dependencies
   └── Documentation.md        # Documentation (legacy)

Base Class Structure
--------------------

All feature selectors inherit from :class:`BaseFeatureSelector` and implement these methods:

- :meth:`forward` - Apply gates to features
- :meth:`regularization_loss` - Compute sparsity penalty
- :meth:`get_selection_probs` - Get feature importance scores
- :meth:`get_selected_features` - Binary feature selection

Class Hierarchy
---------------

::

   BaseFeatureSelector (ABC)
   ├── STGLayer
   ├── STELayer
   ├── GumbelLayer
   ├── CorrelatedSTGLayer
   └── L1Layer

Method Implementations
----------------------

STG Layer (Stochastic Gates)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** :mod:`mylib.selectors`

Uses Gaussian-based continuous relaxation:

.. math::
   
   z_d = \text{clamp}(\mu_d + \epsilon_d, 0, 1)  \text{ where } \epsilon_d \sim N(0, \sigma^2)
   
   \text{regularization} = \sum \Phi(\mu_d / \sigma)  \text{ where } \Phi \text{ is standard normal CDF}

**Advantages**: 
- Low variance gradients
- Stable feature selection
- Theoretical guarantees

**Parameters:**
- ``sigma`` (float, default=0.5): Noise standard deviation for stochastic gates

STE Layer (Straight-Through Estimator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** :mod:`mylib.selectors`

Uses binary gates with gradient flow through sigmoid:

.. math::
   
   z_{\text{hard}} = (\text{sigmoid}(\text{logits}) > 0.5)
   
   z_{\text{soft}} = z_{\text{hard}} - \text{sigmoid}(\text{logits}).\text{detach}() + \text{sigmoid}(\text{logits})
   
   \text{regularization} = \sum \text{sigmoid}(\text{logits})

**Advantages**:
- Simple implementation
- Exact binary gates
- Fast convergence

Gumbel-Softmax Layer
~~~~~~~~~~~~~~~~~~~~

**File:** :mod:`mylib.selectors`

Uses categorical relaxation:

.. math::
   
   \text{logits} = [\text{logit}_{\text{off}}, \text{logit}_{\text{on}}] \text{ for each feature}
   
   z = \text{Gumbel-Softmax}(\text{logits}, \text{temperature}=1.0, \text{hard}=\text{True})
   
   \text{regularization} = \sum \text{softmax}(\text{logits})[:,1]

**Fixed Implementation:**
- Proper initialization bias toward "off" state (encourages sparsity)
- Correct batch dimension handling with broadcasting
- Temperature annealing support via :meth:`set_temperature`

**Advantages**:
- Principled categorical sampling
- Temperature annealing possible
- Good for discrete optimization

**Parameters:**
- ``temperature`` (float, default=1.0): Gumbel-Softmax temperature parameter
- :meth:`set_temperature(t)` method: Update temperature for annealing schedules

Correlated STG Layer
~~~~~~~~~~~~~~~~~~~~

**File:** :mod:`mylib.selectors`

Extension for handling correlated features:

.. math::
   
   \text{regularization} = \sum \Phi(\mu_d/\sigma) + \lambda_{\text{group}} \sum |W_{ij}| * (p_i - p_j)^2

**Advantages**:
- Handles redundant features
- Learns correlation structure
- Better for real-world data

**Parameters:**
- ``sigma`` (float, default=0.5): Noise standard deviation
- ``group_penalty`` (float, default=0.1): Correlation penalty strength

L1 Layer (Baseline)
~~~~~~~~~~~~~~~~~~~

**File:** :mod:`mylib.selectors`

Standard baseline:

.. math::
   
   z = \text{learnable\_weights}  \text{ (no thresholding)}
   
   \text{regularization} = \sum |\text{weights}|

**Advantages**:
- Standard baseline
- Continuous relaxation
- Well-studied properties

Training Strategy
-----------------

Two-Optimizer Approach
~~~~~~~~~~~~~~~~~~~~~~

The :class:`FeatureSelectionTrainer` uses separate optimizers for the model and selector:

.. code-block:: python

   optimizer_model = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
   optimizer_selector = Adam(selector.parameters(), lr=0.01)

Higher learning rate for selector helps gates learn faster.

Loss Function
~~~~~~~~~~~~~

.. math::
   
   \text{total\_loss} = \text{classification\_loss} + \lambda * \text{regularization\_loss}

The regularization strength :math:`\lambda` (lambda) controls the trade-off between accuracy and sparsity.

Lambda Selection
~~~~~~~~~~~~~~~~

The benchmark tests multiple lambda values and selects based on:

.. math::
   
   \text{score} = \text{accuracy} - 0.5 * |\text{selected\_features} - \text{target\_features}|

This balances high accuracy with achieving target sparsity.

Early Stopping
~~~~~~~~~~~~~~

- Monitors validation accuracy
- Patience = 50 epochs
- Minimum 100 epochs before stopping
- Restores best model state automatically

