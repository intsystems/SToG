===========
Quick Start
===========

Basic Example
=============

Here's a minimal example to get started with SToG:

.. code-block:: python

   import torch
   import torch.nn as nn
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   
   from mylib import (
       STGLayer, 
       FeatureSelectionTrainer, 
       create_classification_model
   )
   
   # Load and prepare data
   data = load_breast_cancer()
   X = data.data
   y = data.target
   
   scaler = StandardScaler()
   X = scaler.fit_transform(X)
   
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   X_train, X_val, y_train, y_val = train_test_split(
       X_train, y_train, test_size=0.2, random_state=42
   )
   
   # Convert to PyTorch tensors
   X_train = torch.FloatTensor(X_train)
   y_train = torch.LongTensor(y_train)
   X_val = torch.FloatTensor(X_val)
   y_val = torch.LongTensor(y_val)
   X_test = torch.FloatTensor(X_test)
   y_test = torch.LongTensor(y_test)
   
   # Create model and selector
   n_features = X_train.shape[1]
   n_classes = len(set(y.tolist()))
   
   model = create_classification_model(
       input_dim=n_features,
       num_classes=n_classes
   )
   selector = STGLayer(input_dim=n_features, sigma=0.5)
   
   # Create trainer
   trainer = FeatureSelectionTrainer(
       model=model,
       selector=selector,
       criterion=nn.CrossEntropyLoss(),
       lambda_reg=0.05
   )
   
   # Train
   history = trainer.fit(
       X_train, y_train,
       X_val, y_val,
       epochs=300,
       patience=50,
       verbose=True
   )
   
   # Evaluate
   result = trainer.evaluate(X_test, y_test)
   print(f"\nTest Accuracy: {result['test_acc']:.2f}%")
   print(f"Selected Features: {result['selected_count']} / {n_features}")

Comparing Methods
=================

Compare different feature selection methods:

.. code-block:: python

   from mylib import (
       STGLayer, STELayer, GumbelLayer, 
       CorrelatedSTGLayer, L1Layer
   )
   
   methods = {
       'STG': STGLayer(input_dim=n_features, sigma=0.5),
       'STE': STELayer(input_dim=n_features),
       'Gumbel': GumbelLayer(input_dim=n_features, temperature=1.0),
       'CorrelatedSTG': CorrelatedSTGLayer(input_dim=n_features, sigma=0.5),
       'L1': L1Layer(input_dim=n_features),
   }
   
   results = {}
   for name, selector in methods.items():
       # Create fresh model for fair comparison
       model = create_classification_model(
           input_dim=n_features,
           num_classes=n_classes
       )
       
       trainer = FeatureSelectionTrainer(
           model=model,
           selector=selector,
           criterion=nn.CrossEntropyLoss(),
           lambda_reg=0.05
       )
       
       trainer.fit(X_train, y_train, X_val, y_val, epochs=300, verbose=False)
       results[name] = trainer.evaluate(X_test, y_test)
   
   # Display results
   print(f"\n{'Method':<20} {'Accuracy':<12} {'Selected':<12}")
   print('-' * 44)
   for name, result in results.items():
       print(f"{name:<20} {result['test_acc']:>10.2f}% {result['selected_count']:>10} / {n_features}")

Running Benchmarks
==================

Run comprehensive benchmarks on multiple datasets:

.. code-block:: python

   from mylib import ComprehensiveBenchmark, DatasetLoader
   
   # Initialize benchmark
   benchmark = ComprehensiveBenchmark(device='cpu')
   
   # Optionally customize datasets
   loader = DatasetLoader()
   datasets = [
       loader.load_breast_cancer(),
       loader.load_wine(),
       loader.create_synthetic_high_dim(),
       loader.create_synthetic_correlated(),
   ]
   
   # Run benchmark
   benchmark.run_benchmark(datasets)

Hyperparameter Tuning
=====================

Key hyperparameters and their effects:

Lambda (Sparsity Parameter)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Controls the trade-off between accuracy and sparsity:

.. code-block:: python

   lambdas = [0.01, 0.05, 0.1, 0.2, 0.5]
   
   for lam in lambdas:
       selector = STGLayer(input_dim=n_features)
       trainer = FeatureSelectionTrainer(
           model=create_classification_model(n_features, n_classes),
           selector=selector,
           criterion=nn.CrossEntropyLoss(),
           lambda_reg=lam
       )
       trainer.fit(X_train, y_train, X_val, y_val, epochs=300)
       result = trainer.evaluate(X_test, y_test)
       print(f"λ={lam}: Acc={result['test_acc']:.2f}%, Selected={result['selected_count']}")

Sigma (STG Noise)
~~~~~~~~~~~~~~~~~

Standard deviation of Gaussian noise in STG. Affects smoothness of gates:

.. code-block:: python

   sigmas = [0.1, 0.3, 0.5, 0.8, 1.0]
   
   for sigma in sigmas:
       selector = STGLayer(input_dim=n_features, sigma=sigma)
       # ... train and evaluate ...

Learning Rates
~~~~~~~~~~~~~~

Model and selector have different learning rates:

.. code-block:: python

   trainer = FeatureSelectionTrainer(...)
   # Default: model_lr=0.001, selector_lr=0.01
   # Modify in trainer.__init__ if needed
