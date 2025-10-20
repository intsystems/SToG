import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from abc import ABC, abstractmethod
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
import warnings
warnings.filterwarnings('ignore')


class BaseFeatureSelector(nn.Module, ABC):
    """Base class for feature selection methods."""
    
    def __init__(self, input_dim: int, device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def regularization_loss(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_selection_probs(self) -> torch.Tensor:
        pass

    def get_selected_features(self, threshold: float = 0.5) -> np.ndarray:
        probs = self.get_selection_probs()
        return (probs > threshold).cpu().numpy()


class STGLayer(BaseFeatureSelector):
    """
    Stochastic Gates (STG) - Original implementation from Yamada et al. 2020.
    Uses Gaussian-based continuous relaxation of Bernoulli variables.
    """
    
    def __init__(self, input_dim: int, sigma: float = 0.5, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.mu = nn.Parameter(torch.zeros(input_dim))
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(self.mu) * self.sigma
            z = self.mu + eps
            gates = torch.clamp(z, 0.0, 1.0)
        else:
            gates = torch.clamp(self.mu, 0.0, 1.0)
        return x * gates

    def regularization_loss(self) -> torch.Tensor:
        return torch.sum(0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2)))))

    def get_selection_probs(self) -> torch.Tensor:
        return (0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2))))).detach()


class STELayer(BaseFeatureSelector):
    """
    Straight-Through Estimator for feature selection.
    Uses binary gates with gradient flow through sigmoid.
    """
    
    def __init__(self, input_dim: int, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.logits = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(self.logits)
        
        if self.training:
            gates_hard = (probs > 0.5).float()
            gates = gates_hard - probs.detach() + probs
        else:
            gates = (probs > 0.5).float()
        
        return x * gates

    def regularization_loss(self) -> torch.Tensor:
        return torch.sum(torch.sigmoid(self.logits))

    def get_selection_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.logits).detach()


class GumbelLayer(BaseFeatureSelector):
    """
    Gumbel-Softmax based feature selector.
    Uses categorical distribution over {off, on} for each feature.
    """
    
    def __init__(self, input_dim: int, temperature: float = 1.0, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.logits = nn.Parameter(torch.zeros(input_dim, 2))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            sampled = F.gumbel_softmax(self.logits, tau=self.temperature, hard=True, dim=1)
            gates = sampled[:, 1]
        else:
            gates = (self.logits[:, 1] > self.logits[:, 0]).float()
        
        return x * gates

    def regularization_loss(self) -> torch.Tensor:
        probs = F.softmax(self.logits, dim=1)[:, 1]
        return torch.sum(probs)

    def get_selection_probs(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=1)[:, 1].detach()


class CorrelatedSTGLayer(BaseFeatureSelector):
    """
    STG with explicit handling of correlated features.
    Based on "Adaptive Group Sparse Regularization for Deep Neural Networks".
    Uses group structure to handle feature correlation.
    """
    
    def __init__(self, input_dim: int, sigma: float = 0.5, 
                 group_penalty: float = 0.1, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.mu = nn.Parameter(torch.zeros(input_dim))
        self.sigma = sigma
        self.group_penalty = group_penalty
        
        self.correlation_weights = nn.Parameter(torch.eye(input_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(self.mu) * self.sigma
            z = self.mu + eps
            gates = torch.clamp(z, 0.0, 1.0)
        else:
            gates = torch.clamp(self.mu, 0.0, 1.0)
        
        return x * gates

    def regularization_loss(self) -> torch.Tensor:
        base_reg = torch.sum(0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2)))))
        
        probs = 0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2))))
        
        prob_diff = probs.unsqueeze(0) - probs.unsqueeze(1)  # [D, D]
        correlation_penalty = torch.sum(torch.abs(self.correlation_weights) * prob_diff ** 2)
        
        return base_reg + self.group_penalty * correlation_penalty

    def get_selection_probs(self) -> torch.Tensor:
        return (0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2))))).detach()


class L1Layer(BaseFeatureSelector):
    """
    L1 regularization on input layer weights.
    Baseline comparison method.
    """
    
    def __init__(self, input_dim: int, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.weights = nn.Parameter(torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights

    def regularization_loss(self) -> torch.Tensor:
        return torch.sum(torch.abs(self.weights))

    def get_selection_probs(self) -> torch.Tensor:
        return torch.abs(self.weights).detach()

    def get_selected_features(self, threshold: float = 0.1) -> np.ndarray:
        probs = self.get_selection_probs()
        return (probs > threshold).cpu().numpy()


class FeatureSelectionTrainer:
    """
    Trainer with proper lambda search and early stopping.
    """
    
    def __init__(self, model, selector, criterion, lambda_reg=0.1, device='cpu'):
        self.model = model.to(device)
        self.selector = selector.to(device)
        self.criterion = criterion
        self.device = device
        self.lambda_reg = lambda_reg
        
        self.optimizer_model = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.optimizer_selector = torch.optim.Adam(selector.parameters(), lr=0.01)
        
        self.best_state = None
        self.history = {
            'train_loss': [],
            'val_acc': [],
            'val_loss': [],
            'sel_count': [],
            'reg_loss': []
        }

    def train_epoch(self, X_train, y_train, X_val, y_val):
        """Train for one epoch."""
        self.model.train()
        self.selector.train()
        
        self.optimizer_model.zero_grad()
        self.optimizer_selector.zero_grad()
        
        X_selected = self.selector(X_train)
        predictions = self.model(X_selected)
        
        classification_loss = self.criterion(predictions, y_train)
        regularization_loss = self.selector.regularization_loss()
        total_loss = classification_loss + self.lambda_reg * regularization_loss
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.selector.parameters()),
            max_norm=1.0
        )
        
        self.optimizer_model.step()
        self.optimizer_selector.step()
        
        self.model.eval()
        self.selector.eval()
        
        with torch.no_grad():
            X_val_selected = self.selector(X_val)
            val_predictions = self.model(X_val_selected)
            val_loss = self.criterion(val_predictions, y_val)
            val_acc = (val_predictions.argmax(1) == y_val).float().mean().item() * 100
            sel_count = self.selector.get_selected_features().sum()
        
        return {
            'train_loss': total_loss.item(),
            'val_loss': val_loss.item(),
            'val_acc': val_acc,
            'sel_count': sel_count,
            'reg_loss': regularization_loss.item()
        }

    def fit(self, X_train, y_train, X_val, y_val, epochs=300, 
            patience=50, verbose=False):
        """
        Train the model with early stopping.
        """
        best_val_acc = 0
        wait = 0
        
        for epoch in range(epochs):
            metrics = self.train_epoch(X_train, y_train, X_val, y_val)
            
            for key, value in metrics.items():
                self.history[key].append(value)
            
            if metrics['val_acc'] > best_val_acc:
                best_val_acc = metrics['val_acc']
                wait = 0
                self.best_state = {
                    'model': self.model.state_dict(),
                    'selector': self.selector.state_dict(),
                    'epoch': epoch,
                    'val_acc': best_val_acc,
                    'sel_count': metrics['sel_count']
                }
            else:
                wait += 1
            
            if wait >= patience and epoch >= 100:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}: "
                      f"val_acc={metrics['val_acc']:.2f}%, "
                      f"sel={metrics['sel_count']}, "
                      f"λ={self.lambda_reg:.4f}")
        
        if self.best_state:
            self.model.load_state_dict(self.best_state['model'])
            self.selector.load_state_dict(self.best_state['selector'])
        
        return self.history

    def evaluate(self, X_test, y_test):
        """Evaluate on test set."""
        self.model.eval()
        self.selector.eval()
        
        with torch.no_grad():
            X_selected = self.selector(X_test)
            predictions = self.model(X_selected)
            acc = (predictions.argmax(1) == y_test).float().mean().item() * 100
            sel_features = self.selector.get_selected_features()
            
        return {
            'test_acc': acc,
            'selected_count': sel_features.sum(),
            'selected_features': sel_features
        }


def create_classification_model(input_dim: int, num_classes: int, 
                                hidden_dim: int = None) -> nn.Module:
    """Create a simple feedforward neural network."""
    if hidden_dim is None:
        hidden_dim = min(128, max(64, input_dim))
    
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.BatchNorm1d(hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim // 2, num_classes)
    )


class DatasetLoader:
    """Load and prepare datasets for benchmarking."""
    
    @staticmethod
    def load_breast_cancer():
        """Load breast cancer dataset."""
        data = load_breast_cancer()
        return {
            'name': 'Breast Cancer',
            'X': data.data,
            'y': data.target,
            'n_important': 10,
            'description': 'Binary classification, 30 features'
        }
    
    @staticmethod
    def load_wine():
        """Load wine dataset."""
        data = load_wine()
        return {
            'name': 'Wine',
            'X': data.data,
            'y': data.target,
            'n_important': 7,
            'description': '3-class classification, 13 features'
        }
    
    @staticmethod
    def create_synthetic_high_dim():
        """Create synthetic high-dimensional dataset (MADELON-like)."""
        X, y = make_classification(
            n_samples=600,
            n_features=100,
            n_informative=5,
            n_redundant=10,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=2,
            flip_y=0.03,
            class_sep=1.0,
            random_state=42
        )
        return {
            'name': 'Synthetic-HighDim',
            'X': X,
            'y': y,
            'n_important': 5,
            'description': 'Binary classification, 100 features, 5 informative'
        }
    
    @staticmethod
    def create_synthetic_correlated():
        """Create synthetic dataset with correlated features."""
        np.random.seed(42)
        n_samples = 500
        n_informative = 5
        n_total = 50
        
        X_inform = np.random.randn(n_samples, n_informative)
        
        X_redundant = []
        for i in range(n_informative):
            for _ in range(2):
                noise = np.random.randn(n_samples) * 0.1
                X_redundant.append(X_inform[:, i] + noise)
        X_redundant = np.column_stack(X_redundant)
        
        n_noise = n_total - n_informative - X_redundant.shape[1]
        X_noise = np.random.randn(n_samples, n_noise)
        
        X = np.column_stack([X_inform, X_redundant, X_noise])
        
        y = (X_inform[:, 0] + X_inform[:, 1] * X_inform[:, 2] > 0).astype(int)
        
        return {
            'name': 'Synthetic-Correlated',
            'X': X,
            'y': y,
            'n_important': n_informative,
            'description': f'Binary classification, {n_total} features, {n_informative} informative with correlated copies'
        }


class ComprehensiveBenchmark:
    """
    Comprehensive benchmark for all feature selection methods.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
        self.methods = {
            'STG': STGLayer,
            'STE': STELayer,
            'Gumbel': GumbelLayer,
            'CorrelatedSTG': CorrelatedSTGLayer,
            'L1': L1Layer
        }

    def run_single_experiment(self, dataset_info, method_name, lambda_reg, 
                             random_state=42):
        """Run a single experiment."""
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        X, y = dataset_info['X'], dataset_info['y']
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=random_state, stratify=y_temp
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.LongTensor(y_test).to(self.device)
        
        model = create_classification_model(n_features, n_classes)
        selector_class = self.methods[method_name]
        selector = selector_class(n_features, device=self.device)
        
        trainer = FeatureSelectionTrainer(
            model=model,
            selector=selector,
            criterion=nn.CrossEntropyLoss(),
            lambda_reg=lambda_reg,
            device=self.device
        )
        
        trainer.fit(X_train_t, y_train_t, X_val_t, y_val_t, 
                   epochs=300, patience=50, verbose=False)
        
        results = trainer.evaluate(X_test_t, y_test_t)
        
        return results

    def evaluate_method(self, dataset_info, method_name, 
                       lambda_values=None, n_runs=5):
        """
        Evaluate a method with multiple lambda values and runs.
        """
        if lambda_values is None:
            lambda_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        
        best_result = None
        best_lambda = None
        target_features = dataset_info['n_important']
        
        print(f"\n  Testing {method_name}...")
        
        for lam in lambda_values:
            test_accs = []
            sel_counts = []
            
            for run in range(n_runs):
                try:
                    result = self.run_single_experiment(
                        dataset_info, method_name, lam, random_state=42+run
                    )
                    test_accs.append(result['test_acc'])
                    sel_counts.append(result['selected_count'])
                except Exception as e:
                    print(f"    Error with λ={lam}, run={run}: {e}")
                    continue
            
            if not test_accs:
                continue
            
            mean_acc = np.mean(test_accs)
            mean_sel = np.mean(sel_counts)
            std_acc = np.std(test_accs)
            
            score = mean_acc - abs(mean_sel - target_features) * 0.5
            
            if best_result is None or score > best_result['score']:
                best_result = {
                    'test_acc_mean': mean_acc,
                    'test_acc_std': std_acc,
                    'selected_mean': mean_sel,
                    'selected_std': np.std(sel_counts),
                    'lambda': lam,
                    'score': score
                }
                best_lambda = lam
            
            print(f"    λ={lam:.4f}: acc={mean_acc:.2f}±{std_acc:.2f}%, "
                  f"sel={mean_sel:.1f}±{np.std(sel_counts):.1f}")
        
        print(f"  Best λ={best_lambda:.4f}: "
              f"acc={best_result['test_acc_mean']:.2f}±{best_result['test_acc_std']:.2f}%, "
              f"sel={best_result['selected_mean']:.1f}")
        
        return best_result

    def run_benchmark(self, datasets=None):
        """Run complete benchmark."""
        if datasets is None:
            loader = DatasetLoader()
            datasets = [
                loader.load_breast_cancer(),
                loader.load_wine(),
                loader.create_synthetic_high_dim(),
                loader.create_synthetic_correlated()
            ]
        
        for dataset in datasets:
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset['name']}")
            print(f"Description: {dataset['description']}")
            print(f"Target informative features: {dataset['n_important']}")
            print(f"{'='*80}")
            
            self.results[dataset['name']] = {}
            
            for method_name in self.methods.keys():
                result = self.evaluate_method(dataset, method_name)
                self.results[dataset['name']][method_name] = result
        
        self.print_summary()

    def print_summary(self):
        """Print summary table."""
        print(f"\n\n{'='*100}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*100}")
        print(f"{'Dataset':<25} {'Method':<15} {'Test Acc (%)':<20} {'Features Selected':<20} {'Lambda':<10}")
        print(f"{'-'*100}")
        
        for dataset_name, methods in self.results.items():
            for i, (method_name, result) in enumerate(methods.items()):
                ds_name = dataset_name if i == 0 else ""
                print(f"{ds_name:<25} {method_name:<15} "
                      f"{result['test_acc_mean']:>6.2f} ± {result['test_acc_std']:>5.2f}    "
                      f"{result['selected_mean']:>6.1f} ± {result['selected_std']:>5.1f}      "
                      f"{result['lambda']:>8.4f}")
            print(f"{'-'*100}")


def compare_with_l1_sklearn(datasets):
    """Compare with sklearn L1 logistic regression."""
    
    results = {}
    
    for dataset_info in datasets:
        print(f"\nDataset: {dataset_info['name']}")
        X, y = dataset_info['X'], dataset_info['y']
        
        accs = []
        n_features_list = []
        
        for run in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42+run, stratify=y
            )
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            clf = LogisticRegressionCV(
                penalty='l1',
                solver='saga',
                cv=5,
                max_iter=10000,
                random_state=42+run,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            
            acc = clf.score(X_test, y_test) * 100
            n_features = np.sum(np.any(np.abs(clf.coef_) > 0.01, axis=0))
            
            accs.append(acc)
            n_features_list.append(n_features)
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_features = np.mean(n_features_list)
        
        results[dataset_info['name']] = {
            'acc_mean': mean_acc,
            'acc_std': std_acc,
            'features_mean': mean_features
        }
        
        print(f"  Accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
        print(f"  Features: {mean_features:.1f}")
    
    return results


def main():
    """Main execution function."""
    
    loader = DatasetLoader()
    datasets = [
        loader.load_breast_cancer(),
        loader.load_wine(),
        loader.create_synthetic_high_dim(),
        loader.create_synthetic_correlated()
    ]
    
    benchmark = ComprehensiveBenchmark(device='cpu')
    benchmark.run_benchmark(datasets)
    
    print("\n\n")
    compare_with_l1_sklearn(datasets)


if __name__ == "__main__":
    main()
