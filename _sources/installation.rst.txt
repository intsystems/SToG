============
Installation
============

Requirements
============

- Python 3.9 or higher
- PyTorch 2.0.0 or higher
- NumPy 1.24.0 or higher
- scikit-learn 1.3.0 or higher

Install from GitHub
====================

Clone the repository:

.. code-block:: bash

   git clone https://github.com/intsystems/SToG.git
   cd SToG

Install the package in development mode:

.. code-block:: bash

   pip install -e .

Install Dependencies
====================

For development with documentation:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -r doc/requirements.txt

For experimentation and benchmarking:

.. code-block:: bash

   pip install jupyter matplotlib seaborn pandas

Verify Installation
===================

Test the installation:

.. code-block:: python

   import torch
   from mylib import STGLayer, FeatureSelectionTrainer, create_classification_model
   
   # Quick smoke test
   model = create_classification_model(n_features=20, n_classes=2)
   selector = STGLayer(n_features=20)
   
   X = torch.randn(32, 20)  # batch_size=32, n_features=20
   y = torch.randint(0, 2, (32,))
   
   X_gated = selector(X)
   print(f"Input shape: {X.shape}")
   print(f"Gated output shape: {X_gated.shape}")
   print("Installation successful!")

Or run tests:

.. code-block:: bash

   python test/run_tests.py check
   python test/run_tests.py all

Building Documentation
======================

To build HTML documentation locally:

.. code-block:: bash

   cd doc
   make html

The documentation will be built in `doc/build/html/`. Open `index.html` in your browser.

Troubleshooting
===============

**Import Error: No module named 'mylib'**

Ensure you installed the package in development mode:

.. code-block:: bash

   cd SToG
   pip install -e .

**PyTorch not found**

Install PyTorch following the official guide at https://pytorch.org/get-started/locally/

For CPU-only installation:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

For CUDA support, refer to the official PyTorch documentation.

**Documentation build fails**

Reinstall documentation dependencies:

.. code-block:: bash

   pip install --upgrade -r doc/requirements.txt
