# Stochastic Gating for Feature Selection

Complete implementation of stochastic gating methods for feature selection in neural networks.

## Overview

This implementation includes all major stochastic gating approaches:

1. **STG (Stochastic Gates)** - Original Gaussian-based method from Yamada et al. 2020
2. **STE (Straight-Through Estimator)** - Binary gates with gradient approximation  
3. **Gumbel-Softmax** - Categorical relaxation for feature gating
4. **Correlated STG** - Extension for handling correlated features
5. **L1 Regularization** - Baseline comparison method

## Architecture

### Base Class Structure

```python
BaseFeatureSelector (ABC)
├── forward() - Apply gates to features
├── regularization_loss() - Compute sparsity penalty
├── get_selection_probs() - Get feature importance scores
└── get_selected_features() - Binary feature selection
```

### Method Implementations

#### 1. STG Layer (Stochastic Gates)

```python
z_d = clamp(μ_d + ε_d, 0, 1)  # where ε_d ~ N(0, σ²)
regularization = sum(Φ(μ_d / σ))  # where Φ is standard normal CDF
```

**Advantages**: 
- Low variance gradients
- Stable feature selection
- Theoretical guarantees

#### 2. STE Layer (Straight-Through Estimator)

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

```python
logits = [logit_off, logit_on] for each feature
z = Gumbel-Softmax(logits, temperature=1.0, hard=True)
regularization = sum(softmax(logits)[:,1])
```

**Advantages**:
- Principled categorical sampling
- Temperature annealing possible
- Good for discrete optimization

#### 4. Correlated STG Layer

```python
regularization = sum(Φ(μ_d/σ)) + λ_group * sum(|W_ij| * (p_i - p_j)²)
```

**Advantages**:
- Handles redundant features
- Learns correlation structure
- Better for real-world data

#### 5. L1 Layer (Baseline)

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

### Quick Test

```bash
python test_stochastic_gating.py
```

### Full Benchmark

```bash
python stochastic_gating_complete.py
```

## Datasets

The benchmark includes:

1. **Breast Cancer** (30 features, binary)
2. **Wine** (13 features, 3-class)
3. **Synthetic High-Dim** (100 features, 5 informative)
4. **Synthetic Correlated** (50 features with redundancy)

## Implementation Details

### Hyperparameters

| Method | Key Parameters | Default Values |
|--------|----------------|----------------|
| STG | σ (noise std) | 0.5 |
| STE | - | - |
| Gumbel | temperature | 1.0 |
| Correlated STG | σ, group_penalty | 0.5, 0.1 |
| L1 | - | - |

## References

1. Yamada et al. (2020). "Feature Selection using Stochastic Gates". ICML.
2. Bengio et al. (2013). "Estimating or Propagating Gradients Through Stochastic Neurons". arXiv.
3. Jang et al. (2017). "Categorical Reparameterization with Gumbel-Softmax". ICLR.
4. Louizos et al. (2017). "Learning Sparse Neural Networks through L0 Regularization". arXiv.

## License

MIT License - Free to use and modify.

## Citation

If you use this implementation, please cite the original papers above.

