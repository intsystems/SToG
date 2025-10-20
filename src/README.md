************
Installation
************

Requirements
============

Installing by using PyPi
========================

Install
-------

Uninstall
---------

Architecture Design
---------

### Class Hierarchy

```
BaseFeatureSelector (ABC)
├── STGLayer
├── STELayer
├── GumbelLayer
├── CorrelatedSTGLayer
└── L1Layer

FeatureSelectionTrainer
├── fit()
├── train_epoch()
├── evaluate()
└── history tracking

ComprehensiveBenchmark
├── run_single_experiment()
├── evaluate_method()
├── run_benchmark()
└── print_summary()
```
## Training Strategy

### Two-Optimizer Approach

```python
optimizer_model = Adam(model.parameters(), lr=0.001)
optimizer_selector = Adam(selector.parameters(), lr=0.01)
```

Higher learning rate for selector helps gates learn faster.

### Loss Function

```python
total_loss = classification_loss + λ * regularization_loss
```

### Lambda Selection

The benchmark tests multiple lambda values and selects based on:
```python
score = accuracy - 0.5 * |selected_features - target_features|
```

This balances high accuracy with achieving target sparsity.

### Early Stopping

- Monitors validation accuracy
- Patience = 50 epochs
- Minimum 100 epochs before stopping
- Restores best model state

## Usage

### Basic Usage

```python
from stochastic_gating_complete import (
    STGLayer, FeatureSelectionTrainer, create_classification_model
)

# Create model and selector
n_features = 100
n_classes = 2
model = create_classification_model(n_features, n_classes)
selector = STGLayer(n_features, sigma=0.5)

# Train
trainer = FeatureSelectionTrainer(
    model=model,
    selector=selector,
    criterion=nn.CrossEntropyLoss(),
    lambda_reg=0.05
)

trainer.fit(X_train, y_train, X_val, y_val, epochs=300)

# Evaluate
result = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {result['test_acc']:.2f}%")
print(f"Selected: {result['selected_count']} features")
```

### Run Full Benchmark

```python
from stochastic_gating_complete import ComprehensiveBenchmark

benchmark = ComprehensiveBenchmark(device='cpu')
benchmark.run_benchmark()  # Tests all methods on multiple datasets
```
