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
