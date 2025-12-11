===========
Quick Start
===========

This section shows a minimal end‑to‑end example using SToG after installing
the package from PyPI:

.. code-block:: bash

   pip install stog

Basic Example: STG on Breast Cancer
===================================

The example below reproduces the main steps from the demo notebook in a
compact form.

.. code-block:: python

   import torch
   import torch.nn as nn
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler

   from SToG import (
       STGLayer,
       FeatureSelectionTrainer,
       create_classification_model,
   )

   # Reproducibility
   torch.manual_seed(42)

   # 1. Load and prepare data
   data = load_breast_cancer()
   X = data.data
   y = data.target

   scaler = StandardScaler()
   X = scaler.fit_transform(X)

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   X_train, X_val, y_train, y_val = train_test_split(
       X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
   )

   # Convert to PyTorch tensors
   X_train = torch.FloatTensor(X_train)
   y_train = torch.LongTensor(y_train)
   X_val = torch.FloatTensor(X_val)
   y_val = torch.LongTensor(y_val)
   X_test = torch.FloatTensor(X_test)
   y_test = torch.LongTensor(y_test)

   # 2. Create model and STG selector
   n_features = X_train.shape[1]
   n_classes = len(torch.unique(y_train))

   model = create_classification_model(
       input_dim=n_features,
       num_classes=n_classes,
   )
   selector = STGLayer(input_dim=n_features, sigma=0.5)

   # 3. Create trainer
   trainer = FeatureSelectionTrainer(
       model=model,
       selector=selector,
       criterion=nn.CrossEntropyLoss(),
       lambda_reg=0.05,
       device="cpu",
   )

   # 4. Train with early stopping
   history = trainer.fit(
       X_train=X_train,
       y_train=y_train,
       X_val=X_val,
       y_val=y_val,
       epochs=300,
       patience=50,
       verbose=True,
   )

   # 5. Evaluate on test set
   result = trainer.evaluate(X_test, y_test)

   print(f"\nTest Accuracy: {result['test_acc']:.2f}%")
   print(f"Selected Features: {result['selected_count']} / {n_features}")

Quick Comparison of Methods
===========================

To mirror the demo, you can quickly compare several selection methods
on the same train/val/test split.

.. code-block:: python

   from SToG import (
       STGLayer,
       STELayer,
       GumbelLayer,
       CorrelatedSTGLayer,
       L1Layer,
       FeatureSelectionTrainer,
       create_classification_model,
   )

   methods = {
       "STG": STGLayer(input_dim=n_features, sigma=0.5),
       "STE": STELayer(input_dim=n_features),
       "Gumbel": GumbelLayer(input_dim=n_features, temperature=1.0),
       "CorrelatedSTG": CorrelatedSTGLayer(input_dim=n_features, sigma=0.5),
       "L1": L1Layer(input_dim=n_features),
   }

   results = {}

   for name, selector in methods.items():
       # fresh model for each method
       model = create_classification_model(
           input_dim=n_features,
           num_classes=n_classes,
       )

       trainer = FeatureSelectionTrainer(
           model=model,
           selector=selector,
           criterion=nn.CrossEntropyLoss(),
           lambda_reg=0.05,
           device="cpu",
       )

       trainer.fit(
           X_train=X_train,
           y_train=y_train,
           X_val=X_val,
           y_val=y_val,
           epochs=300,
           patience=50,
           verbose=False,
       )
       results[name] = trainer.evaluate(X_test, y_test)

   print(f"\n{'Method':<20} {'Accuracy':<12} {'Selected':<12}")
   print("-" * 44)
   for name, res in results.items():
       print(
           f"{name:<20} {res['test_acc']:>10.2f}% "
           f"{res['selected_count']:>10} / {n_features}"
       )

Next Steps
==========

For more advanced usage mirroring the full demo notebook, see:

- :doc:`tutorial` for a detailed synthetic example with plots
- :doc:`api/index` for the full API reference
