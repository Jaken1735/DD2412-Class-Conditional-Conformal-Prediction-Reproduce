import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Verify sys.path
print("PYTHONPATH:", sys.path)

import numpy as np
from conformal.utils import random_split
from conformal.classwise_conformal import run_classwise
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

#### PARAMETERS ####
SEED = 2
N_AVG = 10
num_classes = 100 # CIFAR-100
###################

np.random.seed(SEED)

# SCORING FUNCTIONS
conformal_scores_all = 1 - softmax_scores

# Step 2: Split Data into Calibration and Validation Sets
X_calib, y_calib, X_valid, y_valid = random_split(conformal_scores_all, labels, avg_num_per_class=N_AVG)

# Step 4: Perform Classwise Conformal Prediction
q_hat, predictions = run_classwise(
    calibration_scores=X_calib,
    calibration_labels=y_calib,
    validation_scores=X_valid,
    alpha=0.1,
)


# Step 5: Evaluate the Results
# Compute coverage metrics
coverage_metrics = compute_coverage_metrics(y_valid, predictions, alpha=0.1)
print(f"Coverage Metrics:\n{coverage_metrics}")

# Compute set size metrics
set_size_metrics = compute_set_size_metrics(predictions)
print(f"Set Size Metrics:\n{set_size_metrics}")