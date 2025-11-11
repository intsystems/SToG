Datasets
========

The benchmark includes several datasets for evaluating feature selection methods:

Breast Cancer
-------------

- **Features:** 30
- **Classes:** 2 (binary classification)
- **Target informative features:** ~10
- **Description:** Real-world medical dataset from sklearn

Wine
----

- **Features:** 13
- **Classes:** 3
- **Target informative features:** ~7
- **Description:** Wine quality dataset from sklearn

Synthetic High-Dim
------------------

- **Features:** 100
- **Classes:** 2
- **Informative features:** 5
- **Description:** High-dimensional sparse dataset for testing scalability

Synthetic Correlated
--------------------

- **Features:** 50
- **Classes:** 2
- **Description:** Contains correlated/redundant features for testing correlation handling

Usage
-----

All datasets are loaded through the :class:`DatasetLoader` class:

.. code-block:: python

   from mylib import DatasetLoader
   
   loader = DatasetLoader()
   
   # Load specific dataset
   dataset = loader.load_breast_cancer()
   
   # Access data
   X = dataset['X']
   y = dataset['y']
   name = dataset['name']
   description = dataset['description']
   n_important = dataset['n_important']

Each dataset returns a dictionary with:
- ``X``: Feature matrix (numpy array)
- ``y``: Target labels (numpy array)
- ``name``: Dataset name (string)
- ``description``: Dataset description (string)
- ``n_important``: Target number of informative features (int)

