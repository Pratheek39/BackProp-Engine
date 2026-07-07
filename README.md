# BackProp-Engine

This project is a from-scratch machine learning implementation built for AI2000 (Foundations of Machine Learning).
It combines a custom backpropagation/autograd engine with a mini ML library in a single codebase.
The neural network side focuses on computational graphs, gradient flow, and modular layer/loss design.
The mini library side includes dataset loading, preprocessing, model selection, and classical ML models.(Similar structure to sci-kit learn library.)
The repository is organized to support both experimentation (notebook + scripts) and reusable library components.
Overall, the goal is to show end-to-end understanding of ML fundamentals without relying on high-level frameworks.

The project contains:

- a backpropagation/autograd neural network engine
- a mini ML library (`my_ml_lib`) with datasets, preprocessing, model selection, linear models, and naive Bayes modules

## Repository Layout

- `BackProp Engine and mini ML library/`
  - `my_ml_lib/` - mini ML library package
    - `datasets/` - dataset loaders (`load_spambase`, `load_fashion_mnist`)
    - `preprocessing/` - `StandardScaler`, `PolynomialFeatures`, `GaussianBasisFeatures`
    - `model_selection/` - `train_test_split`, `train_test_val_split`, `KFold`
    - `linear_models/classification/` - `LogisticRegression`, `LinearDiscriminantAnalysis`, optional least-squares/perceptron
    - `linear_models/regression/` - ridge and Bayesian regression scaffolds
    - `naive_bayes/` - Gaussian/Bernoulli naive Bayes scaffolds
    - `nn/` - autograd engine, NN modules, losses, optimizer
  - `create_best_model.py` - returns a softmax-style linear classifier (`784 -> 10`)
  - `run_spam_experiment.py` - spam classification workflow with CV and standardization
  - `visualize.py` - renders autograd computation graph with Graphviz
  - `capstone_showdown.ipynb` - course notebook with experiments/results
  - `saved_models/` - saved model artifacts

## What Is Implemented

### Backprop Engine

- `Value` class in `my_ml_lib/nn/autograd.py` builds a computation graph and supports backward propagation
- core NN components:
  - `Module` base class with parameter registration and state save/load
  - `Linear`, `ReLU`, `Sigmoid`, `Sequential`
  - losses: `BinaryCrossEntropyLoss`, `CrossEntropyLoss`
  - optimizer: `SGD`

### Mini ML Library

- datasets:
  - `load_spambase()` (expects `spambase.data` in local `data/` folder)
  - `load_fashion_mnist()` (CSV-based loader)
- preprocessing:
  - `StandardScaler`
  - feature expansion helpers (`PolynomialFeatures`, `GaussianBasisFeatures`)
- model selection:
  - holdout splits (`train_test_split`, `train_test_val_split`)
  - `KFold` cross-validation splitter
- classification:
  - `LogisticRegression` (IRLS/Newton-style optimization)
  - `LinearDiscriminantAnalysis`
  - additional least-squares/perceptron modules


## Running

### 1) Spam experiment

Make sure dataset file exists at:

- `BackProp Engine and mini ML library/data/spambase.data`

Then run:

```bash
cd "BackProp Engine and mini ML library"
python run_spam_experiment.py
```

### 2) Computation graph visualization

```bash
cd "BackProp Engine and mini ML library"
python visualize.py
```

Expected output artifact:

- `BackProp Engine and mini ML library/example_computation_graph.svg`

### 3) Notebook

```bash
jupyter notebook "BackProp Engine and mini ML library/capstone_showdown.ipynb"
```

## Best Model

- `create_best_model.py` defines `initialize_best_model()`
- model shape: input `(n, 784)` to output `(n, 10)`
- implemented as a linear softmax-style classifier using the custom NN modules

## Notes

- This repository intentionally mixes completed implementations and assignment scaffolds.
- The strongest end-to-end demonstrated path is the custom NN/autograd stack plus logistic-regression-based spam experiment.
