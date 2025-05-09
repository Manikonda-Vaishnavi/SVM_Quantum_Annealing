# Support Vector Machine (SVM) Using Quantum Annealing on D-Wave

---

## Overview

This repository showcases an innovative approach to training Support Vector Machines (SVMs)
using quantum annealing on the D-Wave quantum computer. SVMs are powerful supervised learning
models widely used for classification and regression tasks. Traditional SVM training involves
solving a quadratic optimization problem, which can be computationally intensive for large datasets.

Quantum annealing offers a promising alternative by mapping the SVM optimization problem onto a form solvable
by D-Wave’s quantum hardware. This harnesses quantum effects to potentially find solutions faster or explore complex 
optimization landscapes more effectively than classical methods.

---

## What is Quantum Annealing for SVM?

Quantum annealing is a method designed to find the global minimum of a problem expressed as a Quadratic
Unconstrained Binary Optimization (QUBO) or Binary Quadratic Model (BQM). For SVMs, the key insight is
transforming the SVM’s dual optimization formulation into a QUBO that can be executed on the D-Wave quantum annealer.

By doing this, the D-Wave system samples low-energy states of the problem corresponding to classifiers
with maximal margin and minimized error, allowing the quantum-supported SVM to learn decision boundaries for classification tasks.

---

## D-Wave SVM Code Implementation Explanation

The core steps of implementing SVM with quantum annealing on D-Wave include:

### 1. Data Preparation and Kernel Matrix Computation
- The input dataset (features and labels) is first processed.
- A kernel matrix (e.g., linear or RBF kernel) is computed encoding similarities between data points in the feature space.

### 2. Problem Formulation as a Quadratic Model
- The SVM dual optimization problem is transformed into a Binary Quadratic Model (BQM) or QUBO.
- Each binary variable corresponds to a discretized value of the dual variables (Lagrange multipliers).
- Constraints ensuring correct margin maximization and classification accuracy are incorporated as penalty terms in the model.

### 3. Quantum Sampling
- The constructed BQM is submitted to the D-Wave quantum annealer via its SDK.
- A quantum hybrid sampler such as `LeapHybridSampler` or embedding approach is used to solve large or constrained problems.
- The sampler returns samples from the quantum annealer representing candidate solutions.

### 4. Solution Processing and Classification
- The best (lowest energy) sample is selected.
- Dual variables are extracted and transformed back to continuous values.
- The SVM decision boundary is constructed and tested on the dataset.

---

## Example Code Snippet (Conceptual)

```python
from dwave.system import LeapHybridSampler
import dimod
import numpy as np

# Define data and labels
X = ...
y = ...

# Compute kernel matrix K
K = compute_kernel(X)

# Build QUBO coefficients representing the SVM dual problem
Q = build_svm_qubo(K, y)

# Create Binary Quadratic Model
bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# Use D-Wave system sampler
sampler = LeapHybridSampler()
sampleset = sampler.sample(bqm)

# Extract best solution and evaluate
best_sample = sampleset.first.sample
alphas = decode_solution(best_sample)
evaluate_svm(X, y, alphas)
Benefits and Challenges
Benefits:

Quantum annealing provides a novel computational paradigm for solving large-scale optimization problems inherent
in SVM training. Potential speedups for certain problem structures and improved exploration of solution spaces.
A gateway to leveraging emerging quantum hardware with practical machine learning tasks.

#
Challenges:
Mapping continuous SVM variables onto discrete qubits requires approximation and careful encoding.
Current quantum hardware limits problem size and precision.
Need for hybrid quantum-classical workflows for handling real-world large datasets.

# Conclusion
This project demonstrates a cutting-edge methodology that bridges classical Support Vector Machines
and quantum computing through quantum annealing on D-Wave hardware. By translating the SVM optimization
task into a form solvable by quantum annealers, it opens avenues for faster, potentially more effective
machine learning model training as quantum technologies mature.
