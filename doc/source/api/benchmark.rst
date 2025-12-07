=========
benchmark
=========

Benchmarking Framework
======================

.. automodule:: mylib.benchmark
   :members:
   :undoc-members:
   :show-inheritance:

ComprehensiveBenchmark
======================

.. autoclass:: mylib.benchmark.ComprehensiveBenchmark
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

The :class:`mylib.benchmark.ComprehensiveBenchmark` provides a framework for systematically 
comparing feature selection methods across multiple datasets and hyperparameter settings.

Features
~~~~~~~~

- **Multi-method comparison** - STG, STE, Gumbel, CorrelatedSTG, L1
- **Multiple datasets** - Real and synthetic benchmark datasets
- **Lambda search** - Automatic grid search for optimal sparsity parameter
- **Results aggregation** - Summary statistics and comparison tables
- **Result persistence** - Option to save results for later analysis

Benchmarking Pipeline
~~~~~~~~~~~~~~~~~~~~~

The benchmark runs the following pipeline for each method/dataset combination:

.. code-block:: text

   For each dataset:
       For each lambda in [0.001, 0.01, 0.05, 0.1, 0.2, ...]:
           For each feature selection method:
               1. Create fresh model and selector
               2. Initialize trainer with current λ
               3. Train for up to 300 epochs with early stopping
               4. Evaluate on test set
               5. Record: accuracy, selected count, sparsity
               6. Select best λ by balanced score:
                  score = accuracy - 0.5 * |selected - target|
               7. Report best result

Running Benchmarks
~~~~~~~~~~~~~~~~~~

**Basic Usage:**

.. code-block:: python

   from mylib import ComprehensiveBenchmark
   
   benchmark = ComprehensiveBenchmark(device='cpu')
   benchmark.run_benchmark()  # Uses default datasets

**Custom Datasets:**

.. code-block:: python

   from mylib import DatasetLoader, ComprehensiveBenchmark
   
   loader = DatasetLoader()
   datasets = [
       loader.load_breast_cancer(),
       loader.create_synthetic_high_dim(),
   ]
   
   benchmark = ComprehensiveBenchmark()
   benchmark.run_benchmark(datasets)

**GPU Acceleration:**

.. code-block:: python

   benchmark = ComprehensiveBenchmark(device='cuda')
   benchmark.run_benchmark()

Output Format
~~~~~~~~~~~~~

Benchmark prints results in tabular format:

.. code-block:: text

   ==================== Breast Cancer ====================
   
   Method        | Accuracy  | Selected | Sparsity | Lambda
   ______________|___________|__________|__________|________
   STG           | 95.67 %   | 8 / 30   | 73.3%    | 0.050
   STE           | 95.08 %   | 10 / 30  | 66.7%    | 0.050
   Gumbel        | 96.04 %   | 9 / 30   | 70.0%    | 0.050
   CorrelatedSTG | 96.04 %   | 9 / 30   | 70.0%    | 0.050
   L1            | 94.29 %   | 12 / 30  | 60.0%    | 0.050

Lambda Grid Search
~~~~~~~~~~~~~~~~~~

By default, tests these lambda values:

.. code-block:: python

   lambdas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

For each method/dataset, the benchmark:

1. Trains multiple models with different λ
2. Selects best λ using balanced score
3. Reports results for best λ

**Score Formula:**

.. math::

   \text{score}(\lambda) = \text{accuracy} - 0.5 \cdot |\text{selected\_count} - \text{target\_count}|

This balances:
- **Accuracy:** higher is better (coefficient +1)
- **Sparsity:** lower selected count is better (coefficient -0.5)
- **Bias:** targets approximately target_count features

Lambda Interpretation
~~~~~~~~~~~~~~~~~~~~~

- :math:`\lambda` too small: selects too many features
- :math:`\lambda` optimal: achieves target sparsity with high accuracy
- :math:`\lambda` too large: selects too few features, drops accuracy

Comparison with Scikit-learn L1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mylib.benchmark.compare_with_l1_sklearn

Compares SToG methods against scikit-learn's L1-regularized classifiers:

.. code-block:: python

   from mylib import compare_with_l1_sklearn, DatasetLoader
   
   loader = DatasetLoader()
   datasets = [loader.load_breast_cancer()]
   
   compare_with_l1_sklearn(datasets)

Example: Running Full Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from mylib import ComprehensiveBenchmark, DatasetLoader
   
   # Load datasets
   loader = DatasetLoader()
   datasets = [
       loader.load_breast_cancer(),
       loader.load_wine(),
       loader.create_synthetic_high_dim(),
       loader.create_synthetic_correlated(),
   ]
   
   # Run benchmark
   benchmark = ComprehensiveBenchmark(device='cuda' if torch.cuda.is_available() else 'cpu')
   benchmark.run_benchmark(datasets)
   
   # Also compare with sklearn
   from mylib import compare_with_l1_sklearn
   compare_with_l1_sklearn(datasets)

Interpreting Results
~~~~~~~~~~~~~~~~~~~~

Key metrics to analyze:

**Accuracy:** 
   How well the model generalizes on test set. Should be high.

**Selected Count:**
   Number of features chosen by the selector. 
   - Too low: may lose important information
   - Too high: defeats purpose of feature selection
   - Optimal: depends on problem, typically 10-30% of original

**Sparsity:**
   Percentage of features discarded (1 - selected/total).
   Higher sparsity means more aggressive selection.

**Method Ranking:**
   - STG/CorrelatedSTG: most balanced
   - STE: fastest convergence
   - Gumbel: good for probabilistic interpretation
   - L1: simple baseline
