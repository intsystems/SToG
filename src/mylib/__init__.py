"""Stochastic Gating Library for Feature Selection."""

__version__ = '0.0.1'

from .base import BaseFeatureSelector
from .selectors import (
    STGLayer,
    STELayer,
    GumbelLayer,
    CorrelatedSTGLayer,
    L1Layer
)
from .trainer import FeatureSelectionTrainer
from .models import create_classification_model
from .datasets import DatasetLoader
from .benchmark import ComprehensiveBenchmark, compare_with_l1_sklearn

__all__ = [
    # Base
    'BaseFeatureSelector',
    # Selectors
    'STGLayer',
    'STELayer',
    'GumbelLayer',
    'CorrelatedSTGLayer',
    'L1Layer',
    # Training
    'FeatureSelectionTrainer',
    'create_classification_model',
    # Data
    'DatasetLoader',
    # Benchmarking
    'ComprehensiveBenchmark',
    'compare_with_l1_sklearn',
]
