========
Overview
========

Problem Formulation
===================

Feature selection is the problem of identifying a subset of relevant features for model training. 
In high-dimensional settings, many features may be redundant or irrelevant, leading to:

- Increased computational cost
- Reduced model interpretability
- Risk of overfitting to noise
- Difficulty in feature engineering for domain experts

The feature selection problem can be formulated as follows. Given a dataset 
:math:`\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}` where :math:`\mathbf{x}_i \in \mathbb{R}^d` 
and :math:`y_i` is a target, we seek to learn both a selection mechanism and a predictor model:

.. math::

   \min_{\mathbf{g}, \mathbf{f}} \mathcal{L}(\mathbf{g}, \mathbf{f}) + \lambda \Omega(\mathbf{g})

where:

- :math:`\mathbf{g}(\mathbf{x}) \in \{0, 1\}^d` is a discrete feature selector
- :math:`\mathbf{f}` is the prediction model
- :math:`\mathcal{L}` is the task loss (e.g., classification loss)
- :math:`\Omega(\mathbf{g})` encourages sparsity
- :math:`\lambda` balances accuracy vs. sparsity

Stochastic Gating Approach
===========================

The challenge is that directly optimizing discrete gates :math:`\mathbf{g}` is intractable. 
Stochastic gating methods replace discrete variables with continuous relaxations:

.. math::

   \mathbf{z}_d = \mu_d + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)

The continuous variables :math:`\mathbf{z}` are passed through a smooth mapping to approximate 
binary gates, enabling gradient-based optimization while maintaining the sparsity-inducing property.

Methods Overview
================

Stochastic Gates (STG)
~~~~~~~~~~~~~~~~~~~~~~

Based on Yamada et al. (2020), STG uses Gaussian relaxation to approximate Bernoulli variables.

**Forward pass:**
- Sample from Gaussian: :math:`z_d = \mu_d + \sigma \epsilon_d`
- Apply hard clipping: :math:`\tilde{z}_d = \text{clamp}(z_d, [0, 1])`
- Gate features: :math:`\tilde{\mathbf{x}} = \tilde{\mathbf{z}} \odot \mathbf{x}`

**At inference:** Use deterministic gates :math:`\tilde{z}_d = \text{clamp}(\mu_d, [0, 1])`

**Regularization:** Encourages sparse selection via:

.. math::

   \Omega(\mu, \sigma) = \sum_{d=1}^{D} P(z_d > 0) = \sum_{d=1}^{D} \Phi\left(\frac{\mu_d}{\sigma}\right)

where :math:`\Phi` is the cumulative normal distribution.

Straight-Through Estimator (STE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on Bengio et al. (2013), STE uses binary gates with gradient approximation.

**Forward pass:**
- Compute probabilities: :math:`p_d = \sigma(\text{logit}_d)`
- Hard binarization: :math:`g_d = \begin{cases} 1 & p_d > 0.5 \\ 0 & \text{otherwise} \end{cases}`
- Gate features: :math:`\tilde{\mathbf{x}} = \mathbf{g} \odot \mathbf{x}`

**Gradient approximation:** Straight-through allows backpropagation through the binarization:

.. math::

   \frac{\partial g_d}{\partial \text{logit}_d} \approx \frac{\partial p_d}{\partial \text{logit}_d}

**Regularization:**

.. math::

   \Omega(p) = \sum_{d=1}^{D} p_d

Gumbel-Softmax
~~~~~~~~~~~~~~

Based on Jang et al. (2017), uses categorical distribution relaxation.

**Key idea:** Treat feature selection as categorical choice between {off, on} states.

**Forward pass (training):**
- Apply Gumbel-Softmax with temperature :math:`\tau`
- Use hard=True for discrete samples during forward pass
- Temperature annealing: :math:`\tau \to 0` during training

**Forward pass (inference):** Select argmax state

**Regularization:** Probability of on state:

.. math::

   \Omega = \sum_{d=1}^{D} p(z_{d,\text{on}} = 1 | \text{logits}_d)

Correlated STG
~~~~~~~~~~~~~~

Extension of STG for datasets with correlated features.

**Motivation:** When features are highly correlated, independent selection can lead to 
selecting all correlated copies. CorrelatedSTG uses group regularization.

**Key addition:** Feature correlation structure :math:`\mathbf{C}` informed regularization:

.. math::

   \Omega_{\text{corr}}(\mu, \mathbf{C}) = \sum_{d=1}^{D} \Phi\left(\frac{\mu_d}{\sigma}\right) + 
   \alpha \sum_{d,d'} C_{dd'} \left(\Phi\left(\frac{\mu_d}{\sigma}\right) - \Phi\left(\frac{\mu_{d'}}{\sigma}\right)\right)^2

L1 Regularization
~~~~~~~~~~~~~~~~~

Baseline method using L1 penalty on feature weights.

**Model:** Learn weights :math:`\mathbf{w} \in \mathbb{R}^d` directly:

.. math::

   \tilde{\mathbf{x}} = \mathbf{w} \odot \mathbf{x}

**Regularization:**

.. math::

   \Omega(w) = \sum_{d=1}^{D} |w_d|

**Interpretation:** Weights :math:`\mathbf{w}` indirectly indicate feature importance.

Training Strategy
=================

Joint Optimization
~~~~~~~~~~~~~~~~~~~

The model and feature selector are optimized jointly:

.. math::

   \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(\mathbf{f}, \mathbf{g}) + \lambda \Omega(\mathbf{g})

**Two-optimizer approach:**

- Optimizer 1 (model): lower learning rate (e.g., :math:`\eta_m = 0.001`)
- Optimizer 2 (selector): higher learning rate (e.g., :math:`\eta_s = 0.01`)

Higher selector learning rate allows gates to adapt quickly to the task.

Early Stopping
~~~~~~~~~~~~~~

Training uses validation-based early stopping:

1. Monitor validation accuracy
2. Save best model state when validation metric improves
3. Stop if no improvement for *patience* epochs (default: 50)
4. Requires at least 100 epochs before stopping

Lambda Selection
~~~~~~~~~~~~~~~~

The sparsity parameter :math:`\lambda` controls the trade-off between accuracy and feature count. 
SToG includes grid search for optimal :math:`\lambda`:

.. math::

   \text{score}(\lambda) = \text{accuracy} - 0.5 \cdot |\text{selected\_count} - \text{target\_count}|

This balances achieving target sparsity while maintaining high accuracy.

References
==========

For detailed references, see :doc:`references`.
