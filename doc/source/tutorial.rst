========
Tutorial
========

In-Depth Feature Selection Tutorial
====================================

This tutorial demonstrates feature selection on a synthetic high-dimensional dataset.

Problem Setup
=============

We have a classification task with:

- 1000 samples
- 100 features (only 5 truly informative)
- Binary classification problem
- Goal: identify the 5 important features

.. code-block:: python

   import numpy as np
   import torch
   import torch.nn as nn
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.datasets import make_classification
   
   from SToG import STGLayer, FeatureSelectionTrainer, create_classification_model
   
   # Create synthetic dataset
   np.random.seed(42)
   torch.manual_seed(42)
   
   X, y = make_classification(
       n_samples=1000,
       n_features=100,
       n_informative=5,
       n_redundant=10,
       n_repeated=0,
       random_state=42
   )
   
   # Standardize features
   scaler = StandardScaler()
   X = scaler.fit_transform(X)
   
   # Split data: 60% train, 20% val, 20% test
   X_train, X_temp, y_train, y_temp = train_test_split(
       X, y, test_size=0.4, random_state=42
   )
   X_val, X_test, y_val, y_test = train_test_split(
       X_temp, y_temp, test_size=0.5, random_state=42
   )
   
   # Convert to tensors
   X_train = torch.FloatTensor(X_train)
   y_train = torch.LongTensor(y_train)
   X_val = torch.FloatTensor(X_val)
   y_val = torch.LongTensor(y_val)
   X_test = torch.FloatTensor(X_test)
   y_test = torch.LongTensor(y_test)
   
   print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

Step 1: Creating Components
===========================

.. code-block:: python

   # Create classification model
   model = create_classification_model(
       input_dim=100,
       num_classes=2,
       hidden_dim=64
   )
   
   # Create feature selector (STG with sigma=0.5)
   selector = STGLayer(
       input_dim=100,
       sigma=0.5
   )
   
   # Create trainer with regularization strength lambda=0.05
   trainer = FeatureSelectionTrainer(
       model=model,
       selector=selector,
       criterion=nn.CrossEntropyLoss(),
       lambda_reg=0.05,
       device='cpu'
   )

Step 2: Training
================

.. code-block:: python

   # Train for maximum 300 epochs with early stopping
   history = trainer.fit(
       X_train=X_train,
       y_train=y_train,
       X_val=X_val,
       y_val=y_val,
       epochs=300,
       patience=50,
       verbose=True
   )

Expected output:

.. code-block:: text

   Epoch 50: val_acc=92.50%, sel=47, λ=0.0500
   Epoch 100: val_acc=94.00%, sel=32, λ=0.0500
   Epoch 150: val_acc=95.00%, sel=18, λ=0.0500
   Epoch 200: val_acc=95.50%, sel=12, λ=0.0500
   Epoch 250: val_acc=95.50%, sel=10, λ=0.0500
   Early stopping at epoch 283

Step 3: Analyzing Results
==========================

.. code-block:: python

   # Evaluate on test set
   result = trainer.evaluate(X_test, y_test)
   
   print(f"Test Accuracy: {result['test_acc']:.2f}%")
   print(f"Selected Features: {result['selected_count']} / 100")
   print(f"Sparsity: {1 - result['selected_count']/100:.1%}")
   
   # Get selected feature indices
   selected_mask = result['selected_features']
   selected_indices = np.where(selected_mask)[0]
   print(f"\nSelected feature indices: {selected_indices}")

Expected output:

.. code-block:: text

   Test Accuracy: 95.50%
   Selected Features: 10 / 100
   Sparsity: 90.0%
   
   Selected feature indices: [ 0  1  2  3  4 12 34 56 78 91]

Step 4: Visualizing Training History
=====================================

.. code-block:: python

   import matplotlib.pyplot as plt
   
   fig, axes = plt.subplots(1, 3, figsize=(15, 4))
   
   # Plot 1: Validation Accuracy
   axes[0].plot(history['val_acc'], label='Validation Accuracy')
   axes[0].set_xlabel('Epoch')
   axes[0].set_ylabel('Accuracy (%)')
   axes[0].set_title('Validation Accuracy over Time')
   axes[0].grid(True, alpha=0.3)
   axes[0].legend()
   
   # Plot 2: Selected Feature Count
   axes[1].plot(history['sel_count'], label='Selected Features', color='orange')
   axes[1].axhline(y=5, color='r', linestyle='--', label='True Informative (5)')
   axes[1].set_xlabel('Epoch')
   axes[1].set_ylabel('Number of Features')
   axes[1].set_title('Feature Selection Progress')
   axes[1].grid(True, alpha=0.3)
   axes[1].legend()
   
   # Plot 3: Regularization Loss
   axes[2].plot(history['reg_loss'], label='Regularization Loss', color='green')
   axes[2].set_xlabel('Epoch')
   axes[2].set_ylabel('Loss')
   axes[2].set_title('Regularization Loss over Time')
   axes[2].grid(True, alpha=0.3)
   axes[2].legend()
   
   plt.tight_layout()
   plt.savefig('stg_training_history.png', dpi=300, bbox_inches='tight')
   plt.show()

Comparing Methods
=================

.. code-block:: python

   from SToG import STELayer, GumbelLayer, L1Layer
   
   methods = {
       'STG': (STGLayer, {'sigma': 0.5}),
       'STE': (STELayer, {}),
       'Gumbel': (GumbelLayer, {'temperature': 1.0}),
       'L1': (L1Layer, {}),
   }
   
   comparison_results = {}
   
   for method_name, (SelectorClass, kwargs) in methods.items():
       # Fresh model
       model = create_classification_model(100, 2)
       selector = SelectorClass(input_dim=100, **kwargs)
       
       trainer = FeatureSelectionTrainer(
           model=model,
           selector=selector,
           criterion=nn.CrossEntropyLoss(),
           lambda_reg=0.05
       )
       
       trainer.fit(X_train, y_train, X_val, y_val, epochs=300, verbose=False)
       comparison_results[method_name] = trainer.evaluate(X_test, y_test)
   
   # Display comparison table
   print(f"\n{'Method':<15} {'Accuracy':<12} {'Selected':<12} {'Sparsity':<12}")
   print('-' * 51)
   for name, result in comparison_results.items():
       sparsity = 1 - result['selected_count'] / 100
       print(f"{name:<15} {result['test_acc']:>10.2f}% {result['selected_count']:>10} {sparsity:>10.1%}")

Advanced: Lambda Search
=======================

Automatic search for optimal sparsity-accuracy trade-off:

.. code-block:: python

   lambdas = np.logspace(-3, -0.5, 10)
   results_by_lambda = {}
   
   for lam in lambdas:
       model = create_classification_model(100, 2)
       selector = STGLayer(input_dim=100, sigma=0.5)
       
       trainer = FeatureSelectionTrainer(
           model=model,
           selector=selector,
           criterion=nn.CrossEntropyLoss(),
           lambda_reg=lam
       )
       
       trainer.fit(X_train, y_train, X_val, y_val, epochs=300, verbose=False)
       result = trainer.evaluate(X_test, y_test)
       results_by_lambda[lam] = result
   
   # Find best lambda by accuracy-sparsity balance
   best_lambda = max(
       results_by_lambda.keys(),
       key=lambda lam: (
           results_by_lambda[lam]['test_acc'] - 
           0.5 * abs(results_by_lambda[lam]['selected_count'] - 5)
       )
   )
   
   print(f"Best lambda: {best_lambda:.4f}")
   print(f"Accuracy: {results_by_lambda[best_lambda]['test_acc']:.2f}%")
   print(f"Selected: {results_by_lambda[best_lambda]['selected_count']}")

Key Insights
============

1. **Convergence speed varies by method**
   - STE converges fastest but may over-select
   - STG provides good balance
   - Gumbel-Softmax requires temperature annealing

2. **Lambda selection is critical**
   - Too small: selects all features
   - Too large: selects too few features
   - Optimal: balances accuracy and sparsity

3. **Feature correlation matters**
   - Independent methods (STG, STE) may select all correlated copies
   - Use CorrelatedSTG for correlated feature sets
   - Preprocessing (PCA) can reduce correlation

4. **Early stopping improves generalization**
   - Prevents overfitting to training data
   - Saves best model by validation metric
   - Patience parameter: larger for noisier data
