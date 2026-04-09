# BackProp-Engine

A from scratch implementation of a backpropagation neural network engine as part of the AI2000 (Foundations of Machine Learning) course project.

## Folder Structure

### `/BackProp Engine/`
Main project directory containing:

- **`my_ml_lib/`** - Core machine learning library
  - `nn/` - Neural network module with autograd (automatic differentiation)
    - `autograd.py` - Value class implementing the computational graph and backpropagation
    - `modules/` - Neural network building blocks (Linear, Sequential, etc.)
  - `naive_bayes/` - Naive Bayes classifiers (Gaussian and Bernoulli implementations)
  - `linear_models/` - Linear regression models (Ridge Regression)
  - `preprocessing/` - Feature preprocessing (Gaussian RBF features)
  - `datasets/` - Dataset utilities
  - `model_selection/` - Model validation and selection tools

- **`create_best_model.py`** - Implementation of the best performing model (Softmax classifier)
  - Takes input dimensions (n, 784) for 28x28 pixel images
  - Outputs 10 classes for digit classification

- **`visualize.py`** - Computation graph visualization
  - `draw_dot()` function for visualizing the autograd computation graph
  - Generates SVG/PNG visualizations of the forward and backward pass

- **`run_spam_experiment.py`** - Spam classification experiment and results

- **`capstone_showdown.ipynb`** - Main notebook containing comprehensive results and demonstrations
  - Expected dataset output: (200, 100)
  - Best model performance metrics

## Key Features

- **Custom Autograd Engine** - Manual implementation of automatic differentiation
- **Backpropagation** - Full backpropagation algorithm for neural networks
- **Multiple Classifiers** - Naive Bayes, Ridge Regression, and Neural Networks
- **Visualization** - Computation graph visualization for understanding forward/backward passes