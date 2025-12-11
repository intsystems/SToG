====
main
====

Main Execution Script
=====================

.. automodule:: SToG.main
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The :mod:`SToG.main` module provides the entry point for running complete feature selection 
benchmarks with all implemented methods.

Main Function
~~~~~~~~~~~~~

.. autofunction:: SToG.main.main

The :func:`SToG.main.main` function:

1. Loads all available benchmark datasets
2. Initializes the comprehensive benchmark
3. Runs feature selection with all methods
4. Compares results with scikit-learn L1 regularization
5. Prints summary statistics

Running from Command Line
~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the entire benchmarking pipeline:

.. code-block:: bash

   python -m SToG.main

Or from Python:

.. code-block:: python

   from SToG.main import main
   main()

What Gets Executed
~~~~~~~~~~~~~~~~~~~

1. **Load Datasets**
   - Breast Cancer (UCI dataset)
   - Wine (UCI dataset)
   - Synthetic High-Dimensional (MADELON-like)
   - Synthetic Correlated (custom)

2. **Run Benchmarks**
   - Tests all 5 methods (STG, STE, Gumbel, CorrelatedSTG, L1)
   - Searches optimal lambda for each method
   - Reports accuracy and sparsity

3. **Compare with Scikit-learn**
   - Tests LogisticRegression with L1 penalty
   - Compares feature selection results

4. **Print Summary**
   - Tabular results for each dataset
   - Method rankings by accuracy
   - Recommendations for different use cases

Output Example
~~~~~~~~~~~~~~

.. code-block:: text

   ======================================================================
                        Breast Cancer Dataset
   ======================================================================
   
   STG        | Accuracy: 95.67% | Selected:  8/30 (73.3% sparse)
   STE        | Accuracy: 95.08% | Selected: 10/30 (66.7% sparse)
   Gumbel     | Accuracy: 96.04% | Selected:  9/30 (70.0% sparse)
   Correlated | Accuracy: 96.04% | Selected:  9/30 (70.0% sparse)
   L1         | Accuracy: 94.29% | Selected: 12/30 (60.0% sparse)
   sklearn L1 | Accuracy: 94.50% | Selected: 14/30 (53.3% sparse)
   
   ======================================================================
                    Synthetic High-Dimensional Dataset
   ======================================================================
   
   STG        | Accuracy: 98.33% | Selected:  7/100 (93.0% sparse)
   STE        | Accuracy: 97.50% | Selected:  8/100 (92.0% sparse)
   Gumbel     | Accuracy: 98.33% | Selected:  7/100 (93.0% sparse)
   Correlated | Accuracy: 98.33% | Selected:  6/100 (94.0% sparse)
   L1         | Accuracy: 96.67% | Selected: 12/100 (88.0% sparse)
   sklearn L1 | Accuracy: 96.00% | Selected: 15/100 (85.0% sparse)

Customizing Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

To modify benchmarking behavior, edit or extend the main function:

.. code-block:: python

   from SToG import ComprehensiveBenchmark, DatasetLoader
   
   def custom_benchmark():
       """Custom benchmarking with specific settings."""
       loader = DatasetLoader()
       
       # Select specific datasets
       datasets = [
           loader.create_synthetic_high_dim(),
           loader.create_synthetic_correlated(),
       ]
       
       # Run with GPU if available
       import torch
       device = 'cuda' if torch.cuda.is_available() else 'cpu'
       
       benchmark = ComprehensiveBenchmark(device=device)
       benchmark.run_benchmark(datasets)
   
   if __name__ == '__main__':
       custom_benchmark()

Performance Tips
~~~~~~~~~~~~~~~~

- **Use GPU** for faster training: ``device='cuda'``
- **Reduce epochs** for quick testing: modify trainer defaults
- **Subset datasets** for quick validation: load only 1-2 datasets
- **Parallel processing** would require modifying benchmark class
