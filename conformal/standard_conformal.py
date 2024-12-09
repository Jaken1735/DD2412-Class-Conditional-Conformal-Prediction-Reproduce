import numpy as np
from conformal.metrics import *
from conformal.utils import *

"""
Standard Conformal Prediction Case
"""

def compute_exact_coverage_parameters(scores, alpha, default_value=np.inf):

    n = len(scores)
    if n == 0:
        print(f"Insufficient samples (n={n}). Using default threshold: {default_value}")
        return {'q_a': default_value, 'q_b': default_value, 'gamma': 1.0}

    # Sort the scores in ascending order
    sorted_scores = np.sort(scores)

    # Compute k, the largest integer such that k / (n + 1) <= 1 - alpha
    k = int(np.floor((n + 1) * (1 - alpha)))

    # Handle edge cases where k is 0 or n
    if k == 0:
        q_a = sorted_scores[0]
        q_b = q_a
        gamma = 1.0
    elif k >= n:
        q_a = sorted_scores[-1]
        q_b = q_a
        gamma = 1.0
    else:
        q_a = sorted_scores[k - 1]
        q_b = sorted_scores[k]
        gamma = (n + 1) * (1 - alpha) - k

    exact_params = {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
    return exact_params


def construct_exact_prediction_sets(scores_all, exact_params, seed=0):

    q_a = exact_params['q_a']
    q_b = exact_params['q_b']
    gamma = exact_params['gamma']

    num_samples = len(scores_all)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate random uniform variables for each sample
    U = np.random.uniform(0, 1, size=num_samples)

    prediction_sets = []
    for i in range(num_samples):
        if U[i] <= gamma:
            threshold = q_a
        else:
            threshold = q_b
        prediction_set = np.where(scores_all[i] <= threshold)[0]
        prediction_sets.append(prediction_set)

    return prediction_sets



def compute_conformal_threshold(scores, alpha, default_value=np.inf):
    n_samples = len(scores)

    if n_samples == 0:
        print(f"Insufficient samples (n={n_samples}). Using default threshold: {default_value}")
        return default_value

    quantile_level = np.ceil((n_samples + 1) * (1 - alpha)) / n_samples

    if quantile_level > 1:
        print(f"Quantile level exceeds 1. Using default threshold: {default_value}")
        threshold = default_value
    else:
        threshold = np.quantile(scores, quantile_level, interpolation='higher')

    return threshold


def calculate_q_hat(scores_all, true_labels, alpha):
    if scores_all.ndim == 2:
        indices = np.arange(len(true_labels))
        true_scores = scores_all[indices, true_labels]
    else:
        true_scores = scores_all

    # Compute the conformal threshold
    q_hat = compute_conformal_threshold(true_scores, alpha)

    return q_hat


def generate_prediction_sets(scores_all, q_hat):
    if not np.isscalar(q_hat):
        raise ValueError("q_hat should be a single numeric value.")
    prediction_sets = [np.where(scores <= q_hat)[0] for scores in scores_all]

    return prediction_sets


def perform_standard_conformal_prediction(cal_scores_all, cal_labels, val_scores_all, alpha):
    q_hat = calculate_q_hat(cal_scores_all, cal_labels, alpha)
    predictions = generate_prediction_sets(val_scores_all, q_hat)
    return q_hat, predictions



