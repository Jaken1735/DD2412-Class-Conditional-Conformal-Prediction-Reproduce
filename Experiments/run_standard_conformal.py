import sys
import os
import argparse
import jax
import jax.numpy as jnp

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from conformal.utils import random_split, compute_APS_scores, get_RAPS_scores_all
from conformal.standard_conformal import performConformalPrediction
from conformal.metrics import compute_coverage_metrics, compute_set_size_metrics

# Load CIFAR-100 Data from .npy files
def load_cifar100_data(scores_file='data/results_scores.npy', labels_file='data/results_labels.npy'):
    # Get the absolute path to the data files
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scores_path = os.path.join(base_dir, scores_file)
    labels_path = os.path.join(base_dir, labels_file)

    softmax_scores = jnp.load(scores_path)
    labels = jnp.load(labels_path)
    return softmax_scores, labels

parser = argparse.ArgumentParser()
parser.add_argument('--N_AVG', type=int, default=10, help='Average number per class for calibration')
parser.add_argument('--score_func', nargs='+', default=['standard'], help='Example: --score_func softmax APS RAPS')
args = parser.parse_args()

#### PARAMETERS ####
num_classes = 100  # CIFAR-100
lmbda = 0.0005
kreg = 50
alpha = 0.1
SEED = 2
key = jax.random.PRNGKey(SEED)
###################

# Load softmax scores
softmax_scores, labels = load_cifar100_data()

# Run for each score_func
for sf in args.score_func:
    if sf == 'softmax':
        conformal_scores_all = 1 - softmax_scores
    elif sf == 'APS':
        conformal_scores_all = compute_APS_scores(softmax_scores)
    elif sf == 'RAPS':
        conformal_scores_all = get_RAPS_scores_all(softmax_scores, args.lmbda, args.kreg)
    else:
        raise ValueError(f"Unknown scoring function: {sf}")

    # Split data
    X_calib, y_calib, X_valid, y_valid = random_split(conformal_scores_all, labels, avg_num_per_class=args.N_AVG)

    # Perform Standard Conformal Prediction
    predictions = performConformalPrediction(
        calScoresAll=X_calib,
        calLabels=y_calib,
        valScoresAll=X_valid,
        alpha=alpha,
    )

    # Compute metrics
    coverage_metrics = compute_coverage_metrics(y_valid, predictions, alpha=alpha)
    set_size_metrics = compute_set_size_metrics(predictions)

    print(f"standard,{sf},{args.N_AVG},{coverage_metrics['coverage']},{coverage_metrics['covGap']},{set_size_metrics['mean_size']},{set_size_metrics['std_size']}")
