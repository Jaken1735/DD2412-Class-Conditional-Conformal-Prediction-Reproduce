import sys
import os
import argparse
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from conformal.utils import random_split, compute_APS_scores, get_RAPS_scores_all, load_cifar100_data
from conformal.classwise_conformal import run_classwise
from conformal.metrics import compute_all_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--N_AVG', type=int, default=10, help='Average number per class for calibration')
parser.add_argument('--score_func', nargs='+', default=['softmax'], help='Example: --score_func softmax APS RAPS')
args = parser.parse_args()

#### PARAMETERS ####
num_classes = 100  # CIFAR-100
lmbda = 0.0005
kreg = 50
alpha = 0.1
SEED = 3
key = jax.random.PRNGKey(SEED)
###################

# Load data
softmax_scores, labels = load_cifar100_data()

# Run for each score_func
for sf in args.score_func:
    if sf == 'softmax':
        conformal_scores_all = 1 - softmax_scores
    elif sf == 'APS':
        conformal_scores_all = compute_APS_scores(softmax_scores)
    elif sf == 'RAPS':
        conformal_scores_all = get_RAPS_scores_all(softmax_scores, lmbda, kreg)
    else:
        raise ValueError(f"Unknown scoring function: {sf}")
    
    # Split data
    X_calib, y_calib, X_valid, y_valid = random_split(conformal_scores_all, labels, avg_num_per_class=args.N_AVG)

    q_hat, predictions = run_classwise(
        calibration_scores=X_calib,
        calibration_labels=y_calib,
        validation_scores=X_valid,
        alpha=alpha,
    )

    # Compute metrics
    coverage_metrics, set_size_metrics = compute_all_metrics(y_valid, predictions, alpha)

    variables = f"classwise,{sf},{args.N_AVG},"
    cov_metrics = f"{coverage_metrics['mean_class_cov_gap']},{coverage_metrics['cov_gap_std']},"
    set_metrics = f"{set_size_metrics['set_size_mean']},{set_size_metrics['set_size_std']}"
    print(variables+cov_metrics+set_metrics)