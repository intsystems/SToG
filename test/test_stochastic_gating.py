import pytest
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from SToG.base import BaseFeatureSelector
from SToG.selectors import (
    STGLayer,
    STELayer,
    GumbelLayer,
    CorrelatedSTGLayer,
    L1Layer
)
from SToG.trainer import FeatureSelectionTrainer
from SToG.models import create_classification_model
from SToG.datasets import DatasetLoader
from SToG.benchmark import ComprehensiveBenchmark, compare_with_l1_sklearn


class TestBasicSelectors:
    """Testing basic functionality of all selectors."""
    
    @pytest.fixture
    def input_config(self):
        return {
            'input_dim': 20,
            'batch_size': 10,
            'device': 'cpu'
        }
    
    @pytest.fixture
    def sample_input(self, input_config):
        torch.manual_seed(42)
        return torch.randn(input_config['batch_size'], input_config['input_dim'])
    
    def test_stg_layer_forward(self, input_config, sample_input):
        """Test forward pass for STG layer."""
        selector = STGLayer(input_config['input_dim'], sigma=0.5, device=input_config['device'])
        
        # Training mode
        selector.train()
        out_train = selector(sample_input)
        assert out_train.shape == sample_input.shape
        assert torch.isfinite(out_train).all()
        
        # Eval mode
        selector.eval()
        out_eval = selector(sample_input)
        assert out_eval.shape == sample_input.shape
        assert torch.isfinite(out_eval).all()
    
    def test_stg_layer_regularization(self, input_config):
        """Test regularization for STG layer."""
        selector = STGLayer(input_config['input_dim'])
        reg = selector.regularization_loss()
        
        assert reg.dim() == 0
        assert reg >= 0
        assert torch.isfinite(reg)
    
    def test_stg_layer_selection_probs(self, input_config):
        """Test getting selection probabilities."""
        selector = STGLayer(input_config['input_dim'])
        probs = selector.get_selection_probs()
        
        assert probs.shape == (input_config['input_dim'],)
        assert (probs >= 0).all() and (probs <= 1).all()
        assert torch.isfinite(probs).all()
    
    def test_stg_layer_selected_features(self, input_config):
        """Test getting binary mask of selected features."""
        selector = STGLayer(input_config['input_dim'])
        selected = selector.get_selected_features(threshold=0.5)
        
        assert selected.shape == (input_config['input_dim'],)
        assert np.isin(selected, [0, 1]).all()
    
    def test_ste_layer_forward(self, input_config, sample_input):
        """Test forward pass for STE layer."""
        selector = STELayer(input_config['input_dim'], device=input_config['device'])
        
        # Training mode - should use straight-through
        selector.train()
        out_train = selector(sample_input)
        assert out_train.shape == sample_input.shape
        
        # Eval mode - should use hard gates
        selector.eval()
        out_eval = selector(sample_input)
        assert out_eval.shape == sample_input.shape
        
        # In eval mode should only have 0 and 1 in gates
        with torch.no_grad():
            probs = torch.sigmoid(selector.logits)
            gates = (probs > 0.5).float()
            expected_out = sample_input * gates
            torch.testing.assert_close(out_eval, expected_out, rtol=1e-5, atol=1e-5)
    
    def test_ste_layer_regularization(self, input_config):
        """Test regularization for STE layer."""
        selector = STELayer(input_config['input_dim'])
        reg = selector.regularization_loss()
        
        assert reg.dim() == 0
        assert reg >= 0
        assert torch.isfinite(reg)
    
    def test_gumbel_layer_forward(self, input_config, sample_input):
        """Test forward pass for Gumbel layer."""
        selector = GumbelLayer(input_config['input_dim'], temperature=1.0, device=input_config['device'])
        
        selector.train()
        out_train = selector(sample_input)
        assert out_train.shape == sample_input.shape
        
        selector.eval()
        out_eval = selector(sample_input)
        assert out_eval.shape == sample_input.shape
    
    def test_gumbel_layer_temperature(self, input_config, sample_input):
        """Test temperature effect on Gumbel layer."""
        selector_low_temp = GumbelLayer(input_config['input_dim'], temperature=0.1)
        selector_high_temp = GumbelLayer(input_config['input_dim'], temperature=5.0)
        
        # At low temperature should have harder distribution
        selector_low_temp.train()
        selector_high_temp.train()
        
        out_low = selector_low_temp(sample_input)
        out_high = selector_high_temp(sample_input)
        
        assert out_low.shape == out_high.shape == sample_input.shape
    
    def test_correlated_stg_layer_forward(self, input_config, sample_input):
        """Test forward pass for Correlated STG layer."""
        selector = CorrelatedSTGLayer(
            input_config['input_dim'], 
            sigma=0.5, 
            group_penalty=0.1,
            device=input_config['device']
        )
        
        out = selector(sample_input)
        assert out.shape == sample_input.shape
        assert torch.isfinite(out).all()
    
    def test_correlated_stg_layer_regularization(self, input_config):
        """Test regularization with correlation penalty."""
        selector = CorrelatedSTGLayer(input_config['input_dim'], group_penalty=0.1)
        reg = selector.regularization_loss()
        
        assert reg.dim() == 0
        assert torch.isfinite(reg)
        # Regularization should be >= base STG regularization
        assert reg >= 0
    
    def test_l1_layer_forward(self, input_config, sample_input):
        """Test forward pass for L1 layer."""
        selector = L1Layer(input_config['input_dim'], device=input_config['device'])
        
        out = selector(sample_input)
        assert out.shape == sample_input.shape
        
        # L1 should apply weights to input
        expected = sample_input * selector.weights
        torch.testing.assert_close(out, expected)
    
    def test_l1_layer_regularization(self, input_config):
        """Test L1 regularization."""
        selector = L1Layer(input_config['input_dim'])
        reg = selector.regularization_loss()
        
        assert reg.dim() == 0
        # L1 norm is always >= 0
        assert reg >= 0
        
        # For ones initialization, L1 norm should equal input_dim
        expected_reg = input_config['input_dim']
        torch.testing.assert_close(reg, torch.tensor(expected_reg, dtype=torch.float32))
    
    def test_l1_layer_custom_threshold(self, input_config):
        """Test custom threshold for L1 layer."""
        selector = L1Layer(input_config['input_dim'])
        selector.weights.data = torch.tensor([0.05, 0.15, 0.25] + [0.01] * (input_config['input_dim'] - 3))
        
        selected_01 = selector.get_selected_features(threshold=0.1)
        selected_02 = selector.get_selected_features(threshold=0.2)
        
        # With threshold 0.1 should select 2 features (0.15 and 0.25)
        assert selected_01.sum() == 2
        # With threshold 0.2 should select 1 feature (0.25)
        assert selected_02.sum() == 1


class TestFeatureSelectionTrainer:
    """Testing feature selection trainer."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple synthetic data."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train = torch.FloatTensor(X[:60])
        y_train = torch.LongTensor(y[:60])
        X_val = torch.FloatTensor(X[60:80])
        y_val = torch.LongTensor(y[60:80])
        X_test = torch.FloatTensor(X[80:])
        y_test = torch.LongTensor(y[80:])
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def test_trainer_initialization(self, simple_data):
        """Test trainer initialization."""
        X_train, y_train, X_val, y_val, X_test, y_test = simple_data
        
        model = create_classification_model(X_train.shape[1], 2, hidden_dim=32)
        selector = STGLayer(X_train.shape[1])
        
        trainer = FeatureSelectionTrainer(
            model=model,
            selector=selector,
            criterion=torch.nn.CrossEntropyLoss(),
            lambda_reg=0.05
        )
        
        assert trainer.model is not None
        assert trainer.selector is not None
        assert trainer.lambda_reg == 0.05
        assert trainer.optimizer_model is not None
        assert trainer.optimizer_selector is not None
    
    def test_trainer_single_epoch(self, simple_data):
        """Test single training epoch."""
        X_train, y_train, X_val, y_val, X_test, y_test = simple_data
        
        model = create_classification_model(X_train.shape[1], 2, hidden_dim=32)
        selector = STGLayer(X_train.shape[1])
        
        trainer = FeatureSelectionTrainer(
            model=model,
            selector=selector,
            criterion=torch.nn.CrossEntropyLoss(),
            lambda_reg=0.05
        )
        
        metrics = trainer.train_epoch(X_train, y_train, X_val, y_val)
        
        assert 'train_loss' in metrics
        assert 'val_loss' in metrics
        assert 'val_acc' in metrics
        assert 'sel_count' in metrics
        assert 'reg_loss' in metrics
        
        assert metrics['train_loss'] > 0
        assert metrics['val_acc'] >= 0 and metrics['val_acc'] <= 100
        assert metrics['sel_count'] >= 0
    
    def test_trainer_fit(self, simple_data):
        """Test full training."""
        X_train, y_train, X_val, y_val, X_test, y_test = simple_data
        
        model = create_classification_model(X_train.shape[1], 2, hidden_dim=32)
        selector = STGLayer(X_train.shape[1])
        
        trainer = FeatureSelectionTrainer(
            model=model,
            selector=selector,
            criterion=torch.nn.CrossEntropyLoss(),
            lambda_reg=0.05
        )
        
        history = trainer.fit(
            X_train, y_train, X_val, y_val,
            epochs=50, patience=20, verbose=False
        )
        
        assert 'train_loss' in history
        assert 'val_acc' in history
        assert len(history['train_loss']) > 0
        assert len(history['val_acc']) > 0
        
        assert trainer.best_state is not None
        assert 'model' in trainer.best_state
        assert 'selector' in trainer.best_state
    
    def test_trainer_early_stopping(self, simple_data):
        """Test early stopping."""
        X_train, y_train, X_val, y_val, X_test, y_test = simple_data
        
        model = create_classification_model(X_train.shape[1], 2, hidden_dim=32)
        selector = STGLayer(X_train.shape[1])
        
        trainer = FeatureSelectionTrainer(
            model=model,
            selector=selector,
            criterion=torch.nn.CrossEntropyLoss(),
            lambda_reg=0.05
        )
        
        history = trainer.fit(
            X_train, y_train, X_val, y_val,
            epochs=500, patience=10, verbose=False
        )
        
        assert len(history['train_loss']) < 500
    
    def test_trainer_evaluate(self, simple_data):
        """Test model evaluation."""
        X_train, y_train, X_val, y_val, X_test, y_test = simple_data
        
        model = create_classification_model(X_train.shape[1], 2, hidden_dim=32)
        selector = STGLayer(X_train.shape[1])
        
        trainer = FeatureSelectionTrainer(
            model=model,
            selector=selector,
            criterion=torch.nn.CrossEntropyLoss(),
            lambda_reg=0.05
        )
        
        trainer.fit(X_train, y_train, X_val, y_val, epochs=30, patience=10, verbose=False)
        
        result = trainer.evaluate(X_test, y_test)
        
        assert 'test_acc' in result
        assert 'selected_count' in result
        assert 'selected_features' in result
        
        assert result['test_acc'] >= 0 and result['test_acc'] <= 100
        assert result['selected_count'] >= 0 and result['selected_count'] <= X_test.shape[1]
        assert result['selected_features'].shape == (X_test.shape[1],)


class TestDatasetLoader:
    """Testing dataset loader."""
    
    def test_load_breast_cancer(self):
        """Test loading Breast Cancer dataset."""
        loader = DatasetLoader()
        data = loader.load_breast_cancer()
        
        assert 'name' in data
        assert 'X' in data
        assert 'y' in data
        assert 'n_important' in data
        assert 'description' in data
        
        assert data['X'].shape[0] == data['y'].shape[0]
        assert data['X'].shape[1] == 30
        assert len(np.unique(data['y'])) == 2
    
    def test_load_wine(self):
        """Test loading Wine dataset."""
        loader = DatasetLoader()
        data = loader.load_wine()
        
        assert data['X'].shape[0] == data['y'].shape[0]
        assert data['X'].shape[1] == 13
        assert len(np.unique(data['y'])) == 3
    
    def test_create_synthetic_high_dim(self):
        """Test creating high-dimensional synthetic dataset."""
        loader = DatasetLoader()
        data = loader.create_synthetic_high_dim()
        
        assert data['X'].shape[1] == 100
        assert data['n_important'] == 5
        assert len(np.unique(data['y'])) == 2
    
    def test_create_synthetic_correlated(self):
        """Test creating dataset with correlated features."""
        loader = DatasetLoader()
        data = loader.create_synthetic_correlated()
        
        assert data['X'].shape[1] == 50
        assert data['n_important'] == 5
        
        corr_matrix = np.corrcoef(data['X'].T)
        # Should have high correlations (> 0.8) between some features
        high_corr = (np.abs(corr_matrix) > 0.8).sum() - data['X'].shape[1]
        assert high_corr > 0


class TestModelCreation:
    """Testing model creation."""
    
    def test_create_classification_model_default(self):
        """Test creating model with default parameters."""
        model = create_classification_model(input_dim=20, num_classes=2)
        
        assert isinstance(model, torch.nn.Sequential)
        
        x = torch.randn(10, 20)
        out = model(x)
        assert out.shape == (10, 2)
    
    def test_create_classification_model_custom_hidden(self):
        """Test creating model with custom hidden_dim."""
        model = create_classification_model(input_dim=50, num_classes=3, hidden_dim=256)
        
        x = torch.randn(5, 50)
        out = model(x)
        assert out.shape == (5, 3)
    
    def test_create_classification_model_multiclass(self):
        """Test creating model for multiclass classification."""
        model = create_classification_model(input_dim=10, num_classes=5)
        
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 5)


class TestComprehensiveBenchmark:
    """Testing benchmark - lightweight tests."""
    
    @pytest.fixture
    def tiny_dataset(self):
        """Create tiny dataset for quick testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=3,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        return {
            'name': 'Tiny-Test',
            'X': X,
            'y': y,
            'n_important': 3,
            'description': 'Tiny test dataset'
        }
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        benchmark = ComprehensiveBenchmark(device='cpu')
        
        assert benchmark.device == 'cpu'
        assert len(benchmark.methods) == 5
        assert 'STG' in benchmark.methods
        assert 'STE' in benchmark.methods
        assert 'Gumbel' in benchmark.methods
        assert 'CorrelatedSTG' in benchmark.methods
        assert 'L1' in benchmark.methods
    
    def test_run_single_experiment(self, tiny_dataset):
        """Test single experiment."""
        benchmark = ComprehensiveBenchmark(device='cpu')
        
        result = benchmark.run_single_experiment(
            tiny_dataset,
            method_name='STG',
            lambda_reg=0.05,
            random_state=42
        )
        
        assert 'test_acc' in result
        assert 'selected_count' in result
        assert 'selected_features' in result
        
        assert result['test_acc'] >= 0 and result['test_acc'] <= 100
        assert result['selected_count'] >= 0
    
    def test_evaluate_method_single_lambda(self, tiny_dataset):
        """Test method evaluation with single lambda value."""
        benchmark = ComprehensiveBenchmark(device='cpu')
        
        result = benchmark.evaluate_method(
            tiny_dataset,
            method_name='STG',
            lambda_values=[0.05],
            n_runs=2
        )
        
        assert result is not None
        assert 'test_acc_mean' in result
        assert 'test_acc_std' in result
        assert 'selected_mean' in result
        assert 'lambda' in result


class TestIntegration:
    """Integration tests on small benchmark."""
    
    def test_end_to_end_stg(self):
        """End-to-end test for STG on simple dataset."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        X, y = make_classification(
            n_samples=150,
            n_features=15,
            n_informative=3,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
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
        
        model = create_classification_model(15, 2, hidden_dim=32)
        selector = STGLayer(15, sigma=0.5)
        
        # Train
        trainer = FeatureSelectionTrainer(
            model=model,
            selector=selector,
            criterion=torch.nn.CrossEntropyLoss(),
            lambda_reg=0.005
        )
        
        history = trainer.fit(
            X_train_t, y_train_t, X_val_t, y_val_t,
            epochs=100, patience=20, verbose=False
        )
        
        # Evaluate
        result = trainer.evaluate(X_test_t, y_test_t)
        
        assert result['test_acc'] > 50
        assert result['selected_count'] < 15
        assert result['selected_count'] >= 1
        
        print(f"\n[Integration Test] STG: Accuracy={result['test_acc']:.2f}%, Selected={result['selected_count']}/15")
    
    def test_compare_all_methods_tiny(self):
        """Compare all methods on tiny dataset."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create dataset
        loader = DatasetLoader()
        dataset = {
            'name': 'Quick-Test',
            'X': np.random.randn(120, 12),
            'y': np.random.randint(0, 2, 120),
            'n_important': 4,
            'description': 'Quick test dataset'
        }
        
        dataset['y'] = (dataset['X'][:, 0] + dataset['X'][:, 1] + 
                       dataset['X'][:, 2] + dataset['X'][:, 3] > 0).astype(int)
        
        benchmark = ComprehensiveBenchmark(device='cpu')
        
        results = {}
        methods_to_test = ['STG', 'STE', 'L1']
        
        for method in methods_to_test:
            result = benchmark.run_single_experiment(
                dataset,
                method_name=method,
                lambda_reg=0.05,
                random_state=42
            )
            results[method] = result
            
            assert result['test_acc'] > 40
            print(f"[Quick Benchmark] {method}: Acc={result['test_acc']:.2f}%, Sel={result['selected_count']}")
        
        assert len(results) == len(methods_to_test)


class TestEdgeCases:
    """Testing edge cases."""
    
    def test_single_feature(self):
        """Test with single feature."""
        selector = STGLayer(input_dim=1)
        x = torch.randn(10, 1)
        out = selector(x)
        assert out.shape == (10, 1)
    
    def test_large_batch(self):
        """Test with large batch."""
        selector = STGLayer(input_dim=10)
        x = torch.randn(1000, 10)
        out = selector(x)
        assert out.shape == (1000, 10)
    
    def test_zero_input(self):
        """Test with zero input."""
        selector = STGLayer(input_dim=10)
        x = torch.zeros(5, 10)
        out = selector(x)
        assert out.shape == (5, 10)
        assert torch.isfinite(out).all()
    
    def test_extreme_lambda(self):
        """Test with extreme lambda values."""
        X = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        X_val = torch.randn(20, 10)
        y_val = torch.randint(0, 2, (20,))
        
        model = create_classification_model(10, 2, hidden_dim=16)
        selector = STGLayer(10)
        
        # Very large lambda
        trainer_high = FeatureSelectionTrainer(
            model=model,
            selector=selector,
            criterion=torch.nn.CrossEntropyLoss(),
            lambda_reg=10.0
        )
        
        metrics_high = trainer_high.train_epoch(X, y, X_val, y_val)
        assert torch.isfinite(torch.tensor(metrics_high['train_loss']))
        
        # Very small lambda
        model2 = create_classification_model(10, 2, hidden_dim=16)
        selector2 = STGLayer(10)
        trainer_low = FeatureSelectionTrainer(
            model=model2,
            selector=selector2,
            criterion=torch.nn.CrossEntropyLoss(),
            lambda_reg=0.0001
        )
        
        metrics_low = trainer_low.train_epoch(X, y, X_val, y_val)
        assert torch.isfinite(torch.tensor(metrics_low['train_loss']))


def test_sklearn_comparison_runs():
    """Test that sklearn comparison works."""
    loader = DatasetLoader()
    datasets = [loader.load_breast_cancer()]
    
    results = compare_with_l1_sklearn(datasets)
    
    assert len(results) > 0
    assert 'Breast Cancer' in results
    assert 'acc_mean' in results['Breast Cancer']
    assert 'features_mean' in results['Breast Cancer']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])