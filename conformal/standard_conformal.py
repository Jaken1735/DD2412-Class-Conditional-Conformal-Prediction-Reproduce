import numpy as np
import matplotlib.pyplot as plt
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



def compute_conformal_threshold(scores, alpha, default_value=np.inf, exact=False):
    n_samples = len(scores)

    if exact:
        # Placeholder for exact coverage computation
        exact_params = compute_exact_coverage_parameters(scores, alpha, default_value)
        return exact_params
    else:
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


def calculate_q_hat(scores_all, true_labels, alpha, exact=False, plot_hist=False):
    if scores_all.ndim == 2:
        indices = np.arange(len(true_labels))
        true_scores = scores_all[indices, true_labels]
    else:
        true_scores = scores_all

    # Compute the conformal threshold
    q_hat = compute_conformal_threshold(true_scores, alpha, exact=exact)

    # Optionally plot the score distribution
    if plot_hist:
        plt.hist(true_scores, bins=30, edgecolor='black')
        plt.title('Conformity Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.show()

    return q_hat


def generate_prediction_sets(scores_all, q_hat, exact=False):
    if exact:
        if not isinstance(q_hat, dict):
            raise ValueError("For exact coverage, q_hat must be a dictionary of parameters.")
        # Implement exact coverage prediction set construction here
        prediction_sets = construct_exact_prediction_sets(scores_all, q_hat)
    else:
        if not np.isscalar(q_hat):
            raise ValueError("q_hat should be a single numeric value.")
        prediction_sets = [np.where(scores <= q_hat)[0] for scores in scores_all]

    return prediction_sets


def perform_standard_conformal_prediction(
    cal_scores_all, cal_labels, val_scores_all, val_labels, alpha, exact_coverage=False
):
    # Compute q_hat using calibration data
    q_hat = calculate_q_hat(cal_scores_all, cal_labels, alpha, exact=exact_coverage)

    # Generate prediction sets for validation data
    predictions = generate_prediction_sets(val_scores_all, q_hat, exact=exact_coverage)

    # Compute evaluation metrics (implement these functions separately)
    #coverage_metrics = compute_coverage_metrics(val_labels, predictions, alpha)
    #set_size_metrics = compute_set_size_metrics(predictions)

    return q_hat, predictions



