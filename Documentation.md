### Project Structure

The core decision was between organizing code by technical layers (like `controllers`, `models`, `services`) or by features and capabilities (like `solvers`, `algorithms`).

| Aspect | Package by Layer (Technical Structure) | **Package by Feature (Domain Structure)** |
| :--- | :--- | :--- |
| **Cohesion & Coupling** | Low cohesion within packages, high coupling between packages | **High cohesion within packages, low coupling between packages** |
| **Modularity** | Monolithic; harder to extract features | **Highly modular; easy to extract or add features** |
| **Findability** | Logic for one feature scattered across multiple folders | **All related code for a feature is in one place** |
| **Team Workflow** | Requires coordination across layers for one feature | **A team can own a full, vertical feature slice** |

Organizing by technical layers can scatter the code for a single feature across many directories. In contrast, a **feature-based structure** keeps all components of a single feature—such as a specific gating algorithm—in one place, creating a highly modular and understandable codebase. This is similar to the "Package by Feature" and "Feature folders" concepts.

### A Proposed Structure of Project

Based on the feature-based philosophy, this is our project structure. 

```
stog/
│
├── src/                          # Primary source code
│   ├── core/                     # Foundational utilities & base classes
│   │   ├── base_solver.py
│   │   └── gates.py              # Base gate classes
│   │
│   ├── solvers/                  # Our implemented algorithms
│   │   ├── __init__.py
│   │   ├── stg_solver.py         # Original Stochastic Gating
│   │   ├── l2_ste_solver.py      # L2 with Straight-Through Estimation
│   │   ├── gumbel_softmax_solver.py
│   │   └── correlated_features_solver.py
│   │
│   └── utils/                    # Project-specific helpers
│       ├── data_loader.py
│       └── metrics.py
│
├── experiments/                  # Jupyter notebooks & experimental scripts
│   ├── 1.0-test.ipynb
│   └── 1.1-smth.ipynb
│
├── tests/                        # Unit and integration tests
│   ├── test_core/
│   └── test_solvers/             # Test each solver in isolation
│
├── docs/                         # Generated documentation
├── data/                         # Datasets (with subfolders 'raw', 'processed')
├── models/                       # Saved model checkpoints
├── pyproject.toml                # Project metadata & dependencies
├── README.md                     # Project overview and how to use the `solvers`
└── requirements.txt              # Python dependencies
```
