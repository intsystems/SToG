# Stochastic Gating for Feature Selection

Complete implementation of stochastic gating methods for feature selection in neural networks.

## Overview

This implementation includes all major stochastic gating approaches:

1. **STG (Stochastic Gates)** - Original Gaussian-based method from Yamada et al. 2020
2. **STE (Straight-Through Estimator)** - Binary gates with gradient approximation  
3. **Gumbel-Softmax** - Categorical relaxation for feature gating (fixed implementation)
4. **Correlated STG** - Extension for handling correlated features
5. **L1 Regularization** - Baseline comparison method

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or uv package manager

### Quick Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   
   Windows:
   ```batch
   venv\Scripts\activate
   ```
   
   Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using uv (faster):
   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

4. **Install package in editable mode:**
   ```bash
   pip install -e .
   ```

## Project Structure

```
SToG/
├── src/
│   └── mylib/              # Main library package
│       ├── __init__.py     # Package exports
│       ├── base.py         # BaseFeatureSelector abstract class
│       ├── selectors.py    # All feature selector implementations
│       ├── trainer.py      # FeatureSelectionTrainer class
│       ├── models.py       # Model creation utilities
│       ├── datasets.py     # Dataset loading utilities
│       ├── benchmark.py    # Benchmarking framework
│       └── main.py         # Main execution script
├── test/
│   └── test_stochastic_gating.py  # Test suite
├── requirements.txt        # Python dependencies
└── Documentation.md        # This file
```

## Architecture

### Base Class Structure

```python
BaseFeatureSelector (ABC)
├── forward() - Apply gates to features
├── regularization_loss() - Compute sparsity penalty
├── get_selection_probs() - Get feature importance scores
└── get_selected_features() - Binary feature selection
```

All feature selectors inherit from `BaseFeatureSelector` and implement these methods.

### Method Implementations

#### 1. STG Layer (Stochastic Gates)

**File:** `src/mylib/selectors.py`

```python
z_d = clamp(μ_d + ε_d, 0, 1)  # where ε_d ~ N(0, σ²)
regularization = sum(Φ(μ_d / σ))  # where Φ is standard normal CDF
```

**Advantages**: 
- Low variance gradients
- Stable feature selection
- Theoretical guarantees

**Parameters:**
- `sigma` (float, default=0.5): Noise standard deviation for stochastic gates

#### 2. STE Layer (Straight-Through Estimator)

**File:** `src/mylib/selectors.py`

```python
z_hard = (sigmoid(logits) > 0.5)
z_soft = z_hard - sigmoid(logits).detach() + sigmoid(logits)
regularization = sum(sigmoid(logits))
```

**Advantages**:
- Simple implementation
- Exact binary gates
- Fast convergence

#### 3. Gumbel-Softmax Layer

**File:** `src/mylib/selectors.py`

```python
logits = [logit_off, logit_on] for each feature
# Initialized with bias toward "off" state for sparsity
z = Gumbel-Softmax(logits, temperature=1.0, hard=True)
regularization = sum(softmax(logits)[:,1])
```

**Fixed Implementation:**
- Proper initialization bias toward "off" state (encourages sparsity)
- Correct batch dimension handling with broadcasting
- Temperature annealing support via `set_temperature()`

**Advantages**:
- Principled categorical sampling
- Temperature annealing possible
- Good for discrete optimization

**Parameters:**
- `temperature` (float, default=1.0): Gumbel-Softmax temperature parameter
- `set_temperature(t)` method: Update temperature for annealing schedules

#### 4. Correlated STG Layer

**File:** `src/mylib/selectors.py`

```python
regularization = sum(Φ(μ_d/σ)) + λ_group * sum(|W_ij| * (p_i - p_j)²)
```

**Advantages**:
- Handles redundant features
- Learns correlation structure
- Better for real-world data

**Parameters:**
- `sigma` (float, default=0.5): Noise standard deviation
- `group_penalty` (float, default=0.1): Correlation penalty strength

#### 5. L1 Layer (Baseline)

**File:** `src/mylib/selectors.py`

```python
z = learnable_weights # (no thresholding)
regularization = sum(|weights|)
```

**Advantages**:
- Standard baseline
- Continuous relaxation
- Well-studied properties

## Training Strategy

### Two-Optimizer Approach

The `FeatureSelectionTrainer` uses separate optimizers for the model and selector:

```python
optimizer_model = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
optimizer_selector = Adam(selector.parameters(), lr=0.01)
```

Higher learning rate for selector helps gates learn faster.

### Loss Function

```python
total_loss = classification_loss + λ * regularization_loss
```

The regularization strength `λ` (lambda) controls the trade-off between accuracy and sparsity.

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
- Restores best model state automatically

## Usage

### Basic Usage

```python
from mylib import (
    STGLayer, 
    FeatureSelectionTrainer, 
    create_classification_model
)
import torch.nn as nn

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
print(f"Selected features: {result['selected_features']}")
```

### Using Different Selectors

```python
from mylib import STGLayer, STELayer, GumbelLayer, CorrelatedSTGLayer, L1Layer

# STG with custom sigma
selector = STGLayer(input_dim=100, sigma=0.5)

# Straight-Through Estimator
selector = STELayer(input_dim=100)

# Gumbel-Softmax with temperature
selector = GumbelLayer(input_dim=100, temperature=1.0)
# Can update temperature during training
selector.set_temperature(0.5)

# Correlated STG
selector = CorrelatedSTGLayer(input_dim=100, sigma=0.5, group_penalty=0.1)

# L1 baseline
selector = L1Layer(input_dim=100)
```

### Load Datasets

```python
from mylib import DatasetLoader

loader = DatasetLoader()

# Load built-in datasets
breast_cancer = loader.load_breast_cancer()
wine = loader.load_wine()
synthetic_high_dim = loader.create_synthetic_high_dim()
synthetic_correlated = loader.create_synthetic_correlated()

# Access data
X = breast_cancer['X']
y = breast_cancer['y']
print(f"Dataset: {breast_cancer['name']}")
print(f"Shape: {X.shape}")
print(f"Description: {breast_cancer['description']}")
```

### Run Full Benchmark

```python
from mylib import ComprehensiveBenchmark, DatasetLoader

# Load datasets
loader = DatasetLoader()
datasets = [
    loader.load_breast_cancer(),
    loader.load_wine(),
    loader.create_synthetic_high_dim(),
    loader.create_synthetic_correlated()
]

# Run benchmark
benchmark = ComprehensiveBenchmark(device='cpu')
benchmark.run_benchmark(datasets)

# Or use default datasets
benchmark.run_benchmark()  # Uses all built-in datasets
```

### Compare with sklearn L1

```python
from mylib import compare_with_l1_sklearn, DatasetLoader

loader = DatasetLoader()
datasets = [
    loader.load_breast_cancer(),
    loader.load_wine(),
]

results = compare_with_l1_sklearn(datasets)
```

### Run Tests

```bash
# Make sure package is installed in editable mode
pip install -e .

# export path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run tests
python3 run_tests.py
```

Output should look like:

```bash
========================================================================= test session starts ==========================================================================
platform linux -- Python 3.10.12, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/gkarp/Desktop/work/SToG
plugins: typeguard-2.13.3, hydra-core-1.3.2, jaxtyping-0.3.2, cov-7.0.0
collected 35 items                                                                                                                                                     

test/test_stochastic_gating.py::TestBasicSelectors::test_stg_layer_forward PASSED                                                                                [  2%]
test/test_stochastic_gating.py::TestBasicSelectors::test_stg_layer_regularization PASSED                                                                         [  5%]
test/test_stochastic_gating.py::TestBasicSelectors::test_stg_layer_selection_probs PASSED                                                                        [  8%]
test/test_stochastic_gating.py::TestBasicSelectors::test_stg_layer_selected_features PASSED                                                                      [ 11%]
test/test_stochastic_gating.py::TestBasicSelectors::test_ste_layer_forward PASSED                                                                                [ 14%]
test/test_stochastic_gating.py::TestBasicSelectors::test_ste_layer_regularization PASSED                                                                         [ 17%]
test/test_stochastic_gating.py::TestBasicSelectors::test_gumbel_layer_forward PASSED                                                                             [ 20%]
test/test_stochastic_gating.py::TestBasicSelectors::test_gumbel_layer_temperature PASSED                                                                         [ 22%]
test/test_stochastic_gating.py::TestBasicSelectors::test_correlated_stg_layer_forward PASSED                                                                     [ 25%]
test/test_stochastic_gating.py::TestBasicSelectors::test_correlated_stg_layer_regularization PASSED                                                              [ 28%]
test/test_stochastic_gating.py::TestBasicSelectors::test_l1_layer_forward PASSED                                                                                 [ 31%]
test/test_stochastic_gating.py::TestBasicSelectors::test_l1_layer_regularization PASSED                                                                          [ 34%]
test/test_stochastic_gating.py::TestBasicSelectors::test_l1_layer_custom_threshold PASSED                                                                        [ 37%]
test/test_stochastic_gating.py::TestFeatureSelectionTrainer::test_trainer_initialization PASSED                                                                  [ 40%]
test/test_stochastic_gating.py::TestFeatureSelectionTrainer::test_trainer_single_epoch PASSED                                                                    [ 42%]
test/test_stochastic_gating.py::TestFeatureSelectionTrainer::test_trainer_fit PASSED                                                                             [ 45%]
test/test_stochastic_gating.py::TestFeatureSelectionTrainer::test_trainer_early_stopping PASSED                                                                  [ 48%]
test/test_stochastic_gating.py::TestFeatureSelectionTrainer::test_trainer_evaluate PASSED                                                                        [ 51%]
test/test_stochastic_gating.py::TestDatasetLoader::test_load_breast_cancer PASSED                                                                                [ 54%]
test/test_stochastic_gating.py::TestDatasetLoader::test_load_wine PASSED                                                                                         [ 57%]
test/test_stochastic_gating.py::TestDatasetLoader::test_create_synthetic_high_dim PASSED                                                                         [ 60%]
test/test_stochastic_gating.py::TestDatasetLoader::test_create_synthetic_correlated PASSED                                                                       [ 62%]
test/test_stochastic_gating.py::TestModelCreation::test_create_classification_model_default PASSED                                                               [ 65%]
test/test_stochastic_gating.py::TestModelCreation::test_create_classification_model_custom_hidden PASSED                                                         [ 68%]
test/test_stochastic_gating.py::TestModelCreation::test_create_classification_model_multiclass PASSED                                                            [ 71%]
test/test_stochastic_gating.py::TestComprehensiveBenchmark::test_benchmark_initialization PASSED                                                                 [ 74%]
test/test_stochastic_gating.py::TestComprehensiveBenchmark::test_run_single_experiment PASSED                                                                    [ 77%]
test/test_stochastic_gating.py::TestComprehensiveBenchmark::test_evaluate_method_single_lambda PASSED                                                            [ 80%]
test/test_stochastic_gating.py::TestIntegration::test_end_to_end_stg PASSED                                                                                      [ 82%]
test/test_stochastic_gating.py::TestIntegration::test_compare_all_methods_tiny PASSED                                                                            [ 85%]
test/test_stochastic_gating.py::TestEdgeCases::test_single_feature PASSED                                                                                        [ 88%]
test/test_stochastic_gating.py::TestEdgeCases::test_large_batch PASSED                                                                                           [ 91%]
test/test_stochastic_gating.py::TestEdgeCases::test_zero_input PASSED                                                                                            [ 94%]
test/test_stochastic_gating.py::TestEdgeCases::test_extreme_lambda PASSED                                                                                        [ 97%]
test/test_stochastic_gating.py::test_sklearn_comparison_runs PASSED                                                                                              [100%]
```

## Datasets

The benchmark includes:

1. **Breast Cancer** (30 features, binary classification)
   - Real-world medical dataset
   - Target: ~10 informative features

2. **Wine** (13 features, 3-class classification)
   - Wine quality dataset
   - Target: ~7 informative features

3. **Synthetic High-Dim** (100 features, 5 informative)
   - High-dimensional sparse dataset
   - Tests scalability

4. **Synthetic Correlated** (50 features with redundancy)
   - Contains correlated/redundant features
   - Tests correlation handling

## Implementation Details

### Hyperparameters

| Method | Key Parameters | Default Values |
|--------|----------------|----------------|
| STG | `sigma` (noise std) | 0.5 |
| STE | - | - |
| Gumbel | `temperature` | 1.0 |
| Correlated STG | `sigma`, `group_penalty` | 0.5, 0.1 |
| L1 | - | - |

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model LR | 0.001 | Learning rate for classification model |
| Selector LR | 0.01 | Learning rate for feature selector |
| Weight Decay | 1e-4 | L2 regularization for model |
| Max Epochs | 300 | Maximum training epochs |
| Patience | 50 | Early stopping patience |
| Min Epochs | 100 | Minimum epochs before early stopping |
| Lambda Range | [0.001, 0.005, 0.01, 0.05, 0.1, 0.5] | Regularization strength values tested |

### Key Implementation Fixes

**GumbelLayer Improvements:**
1. **Initialization Bias**: Logits initialized with bias toward "off" state (`logits[:, 0] = 1.0`) to encourage sparsity
2. **Broadcasting**: Proper tensor broadcasting with `.unsqueeze(0)` for batch dimension
3. **Temperature Control**: Added `set_temperature()` method for annealing schedules

**Code Structure:**
- Modular architecture with separate files for each component
- Clear separation of concerns
- Easy to extend with new methods

## API Reference

### BaseFeatureSelector

Base class for all feature selectors.

**Methods:**
- `forward(x: torch.Tensor) -> torch.Tensor`: Apply gates to input features
- `regularization_loss() -> torch.Tensor`: Compute regularization loss
- `get_selection_probs() -> torch.Tensor`: Get selection probabilities
- `get_selected_features(threshold=0.5) -> np.ndarray`: Get binary feature mask

### FeatureSelectionTrainer

Training utility for joint optimization of model and selector.

**Methods:**
- `fit(X_train, y_train, X_val, y_val, epochs=300, patience=50, verbose=False)`: Train model
- `evaluate(X_test, y_test) -> dict`: Evaluate on test set
- `train_epoch(X_train, y_train, X_val, y_val) -> dict`: Train one epoch

## References

1. Yamada et al. (2020). "Learning Feature Sparse Principal Subspace". ICML 2020.
   - Original STG method

2. Bengio et al. (2013). "Estimating or Propagating Gradients Through Stochastic Neurons". arXiv:1308.3432.
   - Straight-through estimator

3. Jang et al. (2017). "Categorical Reparameterization with Gumbel-Softmax". ICLR 2017.
   - Gumbel-Softmax distribution

4. "Adaptive Group Sparse Regularization for Deep Neural Networks"
   - Correlated feature handling

## License

MIT License - Free to use and modify.

## Citation

If you use this implementation, please cite the original papers referenced above.
