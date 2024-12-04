import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Verify sys.path
print("PYTHONPATH:", sys.path)

import numpy as np
from conformal.utils import random_split, compute_softmax_conformity_scores
from conformal.standard_conformal import perform_standard_conformal_prediction
from conformal.metrics import compute_coverage_metrics, compute_set_size_metrics

# Step 1: Load CIFAR-100 Data from .npy files
def load_cifar100_data(scores_file='data/results_scores.npy', labels_file='data/results_labels.npy'):

    # Get the absolute path to the data files
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scores_path = os.path.join(base_dir, scores_file)
    labels_path = os.path.join(base_dir, labels_file)

    softmax_scores = np.load(scores_path)
    labels = np.load(labels_path)
    return softmax_scores, labels

# Load data
softmax_scores, labels = load_cifar100_data()

# Verify data shapes
print(f"Softmax scores shape: {softmax_scores.shape}")
print(f"Labels shape: {labels.shape}")

# Set number of classes
num_classes = 100

# Step 2: Split Data into Calibration and Validation Sets
X_calib, y_calib, X_valid, y_valid = random_split(
    softmax_scores, labels, avg_num_per_class=50, seed=42
)

# Step 3: Compute Conformity Scores
calib_conformity_scores = compute_softmax_conformity_scores(X_calib)
valid_conformity_scores = compute_softmax_conformity_scores(X_valid)

# Step 4: Perform Standard Conformal Prediction
q_hat, predictions = perform_standard_conformal_prediction(
    cal_scores_all=calib_conformity_scores,
    cal_labels=y_calib,
    val_scores_all=valid_conformity_scores,
    val_labels=y_valid,
    alpha=0.05,
    exact_coverage=False
)

# Step 5: Evaluate the Results
# Compute coverage metrics
coverage_metrics = compute_coverage_metrics(y_valid, predictions, alpha=0.05)
print(f"Coverage Metrics:\n{coverage_metrics}")

# Compute set size metrics
set_size_metrics = compute_set_size_metrics(predictions)
print(f"Set Size Metrics:\n{set_size_metrics}")

