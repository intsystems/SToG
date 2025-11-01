"""Benchmarking utilities for feature selection methods."""
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

from .selectors import STGLayer, STELayer, GumbelLayer, CorrelatedSTGLayer, L1Layer
from .models import create_classification_model
from .trainer import FeatureSelectionTrainer
from .datasets import DatasetLoader


class ComprehensiveBenchmark:
    """
    Comprehensive benchmark for all feature selection methods.
    """
    
    def __init__(self, device='cpu'):
        """
        Initialize benchmark.
        
        Args:
            device: Device to run on ('cpu' or 'cuda')
        """
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
        """
        Run a single experiment.
        
        Args:
            dataset_info: Dictionary with dataset information
            method_name: Name of the method to test
            lambda_reg: Regularization strength
            random_state: Random seed
            
        Returns:
            Dictionary with results
        """
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
            criterion=torch.nn.CrossEntropyLoss(),
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
        
        Args:
            dataset_info: Dictionary with dataset information
            method_name: Name of the method to test
            lambda_values: List of lambda values to try
            n_runs: Number of runs per lambda value
            
        Returns:
            Dictionary with best results
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
        """
        Run complete benchmark.
        
        Args:
            datasets: List of dataset info dictionaries (uses default if None)
        """
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
        """Print summary table of benchmark results."""
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
    """
    Compare with sklearn L1 logistic regression baseline.
    
    Args:
        datasets: List of dataset info dictionaries
        
    Returns:
        Dictionary with sklearn results
    """
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

