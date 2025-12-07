========
datasets
========

Dataset Loading Utilities
==========================

.. automodule:: mylib.datasets
   :members:
   :undoc-members:
   :show-inheritance:

DatasetLoader
=============

.. autoclass:: mylib.datasets.DatasetLoader
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The :class:`mylib.datasets.DatasetLoader` provides static methods to load benchmark datasets 
suitable for feature selection experiments.

Available Datasets
~~~~~~~~~~~~~~~~~~

All loaders return a dictionary with:

.. code-block:: python

   {
       'name': str,              # Dataset name
       'X': ndarray,             # Feature matrix [n_samples, n_features]
       'y': ndarray,             # Target labels [n_samples]
       'n_important': int,       # Number of truly informative features
       'description': str        # Dataset description
   }

Breast Cancer Dataset
~~~~~~~~~~~~~~~~~~~~~

.. automethod:: mylib.datasets.DatasetLoader.load_breast_cancer

**Properties:**
   - 569 samples
   - 30 features
   - Binary classification (malignant vs. benign)
   - ~10 informative features
   - Real-world dataset (UCI repository)

**Use case:** General feature selection testing

.. code-block:: python

   from mylib import DatasetLoader
   data = DatasetLoader.load_breast_cancer()
   print(f"Dataset: {data['name']}")
   print(f"Shape: {data['X'].shape}")
   print(f"Informative features: {data['n_important']}")

Wine Dataset
~~~~~~~~~~~~

.. automethod:: mylib.datasets.DatasetLoader.load_wine

**Properties:**
   - 178 samples
   - 13 features
   - 3-class classification
   - ~7 informative features
   - Real-world dataset (UCI repository)

**Use case:** Multi-class feature selection

.. code-block:: python

   data = DatasetLoader.load_wine()
   print(f"Classes: {len(np.unique(data['y']))}")

Synthetic High-Dimensional Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: mylib.datasets.DatasetLoader.create_synthetic_high_dim

**Properties:**
   - 600 samples
   - 100 features
   - Binary classification
   - 5 informative features
   - 10 redundant features
   - 85 noise features
   - Class imbalance: ~3%

**Use case:** High-dimensional feature selection testing

**Generated via:** scikit-learn's ``make_classification``

.. code-block:: python

   data = DatasetLoader.create_synthetic_high_dim()
   print(f"Sparsity level: {5}/{data['X'].shape[1]}")

Synthetic Correlated Features Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: mylib.datasets.DatasetLoader.create_synthetic_correlated

**Properties:**
   - 500 samples
   - 50 features
   - Binary classification
   - 5 core informative features
   - 10 correlated copies (2 per core feature)
   - 35 noise features
   - Target: :math:`y = \mathbb{1}[x_1 + x_2 \cdot x_3 > 0]`

**Use case:** Testing with correlated features

**Characteristics:**
   - Features 0-4: core informative
   - Features 5-14: noisy copies of core features
   - Features 15-49: pure noise
   - High correlation: 0.9+ between core and copies

.. code-block:: python

   data = DatasetLoader.create_synthetic_correlated()
   # Use with CorrelatedSTGLayer for best results

Full Benchmarking Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mylib import DatasetLoader, ComprehensiveBenchmark
   
   loader = DatasetLoader()
   datasets = [
       loader.load_breast_cancer(),
       loader.load_wine(),
       loader.create_synthetic_high_dim(),
       loader.create_synthetic_correlated(),
   ]
   
   benchmark = ComprehensiveBenchmark()
   benchmark.run_benchmark(datasets)

Data Preprocessing
~~~~~~~~~~~~~~~~~~

Recommended preprocessing pipeline:

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   from sklearn.model_selection import train_test_split
   
   data = DatasetLoader.load_breast_cancer()
   X = data['X']
   y = data['y']
   
   # 1. Standardize features
   scaler = StandardScaler()
   X = scaler.fit_transform(X)
   
   # 2. Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   X_train, X_val, y_train, y_val = train_test_split(
       X_train, y_train, test_size=0.2, random_state=42
   )
   
   # 3. Convert to tensors
   X_train = torch.FloatTensor(X_train)
   y_train = torch.LongTensor(y_train)
