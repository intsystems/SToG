=========
selectors
=========

Feature Selection Methods
==========================

.. automodule:: SToG.selectors
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The :mod:`SToG.selectors` module implements five feature selection methods, each with different 
properties and use cases.

Stochastic Gates (STGLayer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SToG.selectors.STGLayer
   :members:
   :undoc-members:
   :show-inheritance:

**Method:** Gaussian-based continuous relaxation (Yamada et al., 2020)

**When to use:**
   - Balanced accuracy and sparsity
   - Need smooth gradient flow
   - Stable training on most datasets

**Parameters:**
   - ``sigma`` - Standard deviation of Gaussian noise (default: 0.5)
   - Larger sigma: more exploration, potentially less sparse
   - Smaller sigma: deterministic behavior, faster convergence

**Example:**

.. code-block:: python

   from SToG import STGLayer
   selector = STGLayer(input_dim=100, sigma=0.5)

Straight-Through Estimator (STELayer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SToG.selectors.STELayer
   :members:
   :undoc-members:
   :show-inheritance:

**Method:** Binary gates with gradient approximation (Bengio et al., 2013)

**When to use:**
   - Need explicit binary decisions (on/off)
   - Prefer fast convergence
   - Working with small feature sets

**Advantages:**
   - Produces true binary gates at inference
   - Fast training convergence
   - Clear feature selection (no fuzzy boundaries)

**Disadvantages:**
   - Gradient approximation may be biased
   - Can get stuck in local optima
   - May over-select features

**Example:**

.. code-block:: python

   from SToG import STELayer
   selector = STELayer(input_dim=100)

Gumbel-Softmax (GumbelLayer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SToG.selectors.GumbelLayer
   :members:
   :undoc-members:
   :show-inheritance:

**Method:** Categorical distribution relaxation (Jang et al., 2017)

**When to use:**
   - Need principled probabilistic framework
   - Working with discrete latent variables
   - Can afford temperature annealing schedule

**Parameters:**
   - ``temperature`` - Initial temperature (default: 1.0)
   - Temperature annealing: :math:`\tau \to 0` during training
   - Smaller temperature: more discrete behavior

**Advantages:**
   - Theoretically grounded in Gumbel distribution
   - Flexible temperature schedule
   - Good for categorical problems

**Example:**

.. code-block:: python

   from SToG import GumbelLayer
   selector = GumbelLayer(input_dim=100, temperature=1.0)

Correlated STG (CorrelatedSTGLayer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SToG.selectors.CorrelatedSTGLayer
   :members:
   :undoc-members:
   :show-inheritance:

**Method:** STG variant for correlated features

**When to use:**
   - Features have high correlation
   - Want to avoid selecting all correlated copies
   - Need group-aware feature selection

**How it works:**
   - Computes feature correlation structure
   - Adds group regularization penalty
   - Encourages correlated groups to be selected together or not at all

**Parameters:**
   - ``correlation_threshold`` - Threshold for grouping correlated features
   - ``group_alpha`` - Weight of group regularization

**Example:**

.. code-block:: python

   from SToG import CorrelatedSTGLayer
   selector = CorrelatedSTGLayer(
       input_dim=100, 
       sigma=0.5,
       correlation_threshold=0.8
   )

L1 Regularization (L1Layer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SToG.selectors.L1Layer
   :members:
   :undoc-members:
   :show-inheritance:

**Method:** Classical L1 penalty on feature weights

**When to use:**
   - Baseline comparison
   - Want interpretable feature weights
   - Need simple, proven method

**How it works:**
   - Learns feature weights :math:`w \in \mathbb{R}^d`
   - Gates input: :math:`\tilde{x} = w \odot x`
   - Encourages small weights via L1 penalty

**Advantages:**
   - Simple and interpretable
   - Fast convergence
   - Well-studied statistical properties

**Disadvantages:**
   - Soft selection (weights are continuous)
   - May not achieve exact sparsity
   - Features selected by magnitude, not binary gates

**Example:**

.. code-block:: python

   from SToG import L1Layer
   selector = L1Layer(input_dim=100)

Method Comparison
~~~~~~~~~~~~~~~~~

.. list-table:: Feature Selection Methods Comparison
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Convergence
     - Sparsity
     - Interpretability
     - Stability
     - Use Case
   * - STG
     - Medium
     - Good
     - Good
     - High
     - General purpose
   * - STE
     - Fast
     - Good
     - Excellent
     - Medium
     - Binary selection
   * - Gumbel
     - Medium
     - Good
     - Good
     - Medium
     - Categorical
   * - CorrelatedSTG
     - Slow
     - Excellent
     - Good
     - High
     - Correlated features
   * - L1
     - Fast
     - Fair
     - Good
     - High
     - Baseline
