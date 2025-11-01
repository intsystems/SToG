"""Dataset loading utilities."""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, make_classification


class DatasetLoader:
    """Load and prepare datasets for benchmarking."""
    
    @staticmethod
    def load_breast_cancer():
        """
        Load breast cancer dataset.
        
        Returns:
            Dictionary with dataset information
        """
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
        """
        Load wine dataset.
        
        Returns:
            Dictionary with dataset information
        """
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
        """
        Create synthetic high-dimensional dataset (MADELON-like).
        
        Returns:
            Dictionary with dataset information
        """
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
        """
        Create synthetic dataset with correlated features.
        
        Returns:
            Dictionary with dataset information
        """
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

