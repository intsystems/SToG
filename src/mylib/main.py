"""Main execution script for benchmarking."""
from .benchmark import ComprehensiveBenchmark, compare_with_l1_sklearn
from .datasets import DatasetLoader


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

