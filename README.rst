SToG: Stochastic Gates for Feature Selection
============================================

.. figure:: https://raw.githubusercontent.com/intsystems/SToG/master/figures/logo_stog.png
    :width: 400
    :align: center
    :alt: SToG Logo

|test| |docs|

.. |test| image:: https://github.com/intsystems/SToG/workflows/test/badge.svg
    :target: https://github.com/intsystems/SToG/actions/workflows/test.yaml
    :alt: Test status

.. |docs| image:: https://github.com/intsystems/SToG/workflows/docs/badge.svg
    :target: https://intsystems.github.io/SToG/
    :alt: Docs status

**SToG** (Stochastic Gates) is a PyTorch-based library designed for efficient and differentiable feature selection in neural networks. It implements various stochastic gating mechanisms that allow models to learn sparse feature representations end-to-end.

The library includes implementations of:

*   **STG (Stochastic Gates):** Gaussian-based relaxation of Bernoulli gates.
*   **STE (Straight-Through Estimator):** Hard thresholding with gradient approximation.
*   **Gumbel-Softmax:** Categorical reparameterization for feature selection.
*   **Correlated STG:** Handles multi-collinearity among features.
*   **L1 Regularization:** Classic Lasso-style selection layer.

Quick Links
-----------

*   `Documentation <https://intsystems.github.io/SToG/>`_
*   `Installation Guide <https://intsystems.github.io/SToG/installation.html>`_
*   `Quick Start <https://intsystems.github.io/SToG/quick_start.html>`_
*   `API Reference <https://intsystems.github.io/SToG/api/index.html>`_
*   `Blogpost <https://rubtsov-bmm-course.hashnode.dev/stog-library>`_

Installation
------------

You can install the package directly from the source:

.. code-block:: bash

    pip install SToG

Project Information
===================

:Project Title: Stochastic Gating for Robust Feature Selection
:Project Type: Research Project
:Authors: Eynullayev Altay, Firsov Sergey, Rubtsov Denis, Karpeev Gleb

Abstract
========

Feature selection is a crucial step in building interpretable and efficient machine learning models, especially in high-dimensional settings. This project investigates and implements stochastic gating mechanisms—a class of differentiable relaxation methods that enable gradient-based feature selection. 

We provide a comprehensive library **SToG**, which allows researchers and practitioners to easily plug in feature selection layers into existing PyTorch architectures. The library supports various regularization techniques, handles correlated features, and provides a unified interface for benchmarking different selection strategies against standard baselines.
