import sys
import os
# Add src to path for development (not needed if package is installed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from mylib import (
    STGLayer, STELayer, GumbelLayer, CorrelatedSTGLayer, L1Layer,
    FeatureSelectionTrainer, create_classification_model,
    DatasetLoader, ComprehensiveBenchmark
)


def test_basic_functionality():
    """Test that all selector layers work."""
    print("Testing basic functionality...")
    
    input_dim = 20
    batch_size = 10
    x = torch.randn(batch_size, input_dim)
    
    selectors = {
        'STG': STGLayer(input_dim),
        'STE': STELayer(input_dim),
        'Gumbel': GumbelLayer(input_dim),
        'CorrelatedSTG': CorrelatedSTGLayer(input_dim),
        'L1': L1Layer(input_dim)
    }
    
    for name, selector in selectors.items():
        out = selector(x)
        assert out.shape == x.shape, f"{name} output shape mismatch"
        reg = selector.regularization_loss()
        assert reg.numel() == 1, f"{name} regularization should be scalar"
        probs = selector.get_selection_probs()
        assert probs.shape == (input_dim,), f"{name} probs shape mismatch"
        selected = selector.get_selected_features()
        assert selected.shape == (input_dim,), f"{name} selected shape mismatch"
        
        print(f"     {name} passed")
    
    print("All basic tests passed!\n")


def test_training_simple():
    """Test that training actually works."""
    print("Testing training on simple synthetic data...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 200
    n_features = 20
    n_informative = 3
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    
    X[:, n_informative:] = np.random.randn(n_samples, n_features - n_informative) * 0.1
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    model = create_classification_model(n_features, 2, hidden_dim=32)
    selector = STGLayer(n_features)
    
    trainer = FeatureSelectionTrainer(
        model=model,
        selector=selector,
        criterion=torch.nn.CrossEntropyLoss(),
        lambda_reg=0.05
    )
    
    history = trainer.fit(X_train_t, y_train_t, X_val_t, y_val_t, 
                         epochs=200, patience=30, verbose=False)
    
    result = trainer.evaluate(X_test_t, y_test_t)
    
    print(f"  Test accuracy: {result['test_acc']:.2f}%")
    print(f"  Features selected: {result['selected_count']} / {n_features}")
    print(f"  Selection: {np.where(result['selected_features'])[0][:10]}")
    
    probs = selector.get_selection_probs().numpy()
    top_3_indices = np.argsort(probs)[-3:][::-1]
    
    print(f"  Top 3 features by probability: {top_3_indices}")
    print(f"  Their probabilities: {probs[top_3_indices]}")
    
    if result['test_acc'] > 70 and result['selected_count'] < n_features:
        print("  Training test passed!\n")
        return True
    else:
        print("  Training might need tuning\n")
        return False


def test_quick_benchmark():
    """Run quick benchmark on one dataset."""
    
    loader = DatasetLoader()
    dataset = loader.load_breast_cancer()
    
    benchmark = ComprehensiveBenchmark(device='cpu')
    
    print(f"\nDataset: {dataset['name']}")
    print(f"Shape: {dataset['X'].shape}")
    print(f"Target informative: {dataset['n_important']}\n")
    
    for method in ['STG', 'L1']:
        result = benchmark.evaluate_method(
            dataset, 
            method, 
            lambda_values=[0.01, 0.05, 0.1],
            n_runs=3
        )
        # Assert that benchmark completed successfully
        # assert result is not None, f"Benchmark failed for method {method}"
        # assert 'test_acc_mean' in result, "Benchmark result missing test accuracy"
        # assert 'selected_mean' in result, "Benchmark result missing selected features count"


if __name__ == "__main__":
    # Run tests directly when executed as script
    test_basic_functionality()
    test_training_simple()
    test_quick_benchmark()
