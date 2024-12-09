import os
import numpy as np  # Still use NumPy for data loading
import jax
import jax.numpy as jnp


"""
Data Preprocessing utils function below
"""

def random_split(X, y, avg_num_per_class, seed=0):
    #np.random.seed(seed)
    num_classes = np.max(y) + 1
    num_samples = avg_num_per_class * num_classes
    
    idx1 = np.random.choice(np.arange(len(y)), size=num_samples, replace=False)
    idx2 = ~np.isin(np.arange(len(y)), idx1) 
    X1, y1 = X[idx1], y[idx1]
    X2, y2 = X[idx2], y[idx2]
    
    return X1, y1, X2, y2


"""
Function for remapping after filtering out rare classes
"""
def reinitClasses(labels, rare_classes):
    '''
    Exclude classes in rare_classes and remap remaining classes to be 0-indexed.

    Outputs:
        - remaining_idx: Boolean array the same length as labels. Entry i is True
          iff labels[i] is not in rare_classes.
        - remapped_labels: Array that only contains the entries of labels that are 
          not in rare_classes (in order), remapped to 0-based indices.
        - remapping: Dict mapping old class index to new class index.
    '''
    # Identify non-rare samples
    remaining_idx = ~np.isin(labels, rare_classes)
    
    # Extract non-rare labels
    remaining_labels = labels[remaining_idx]
    
    # Use np.unique to get unique labels and remap labels
    unique_labels, remapped_labels = np.unique(remaining_labels, return_inverse=True)
    
    # Create remapping dictionary
    remapping = {original_label: new_label for new_label, original_label in enumerate(unique_labels)}
    
    return remaining_idx, remapped_labels, remapping



"""
Scoring Functions below
"""

def compute_APS_scores(softmax_scores):
    n_samples, n_classes = softmax_scores.shape

    # Step 1: Sort the softmax scores in descending order for each sample
    sorted_indices = jnp.argsort(-softmax_scores, axis=1)
    sorted_softmax = jnp.take_along_axis(softmax_scores, sorted_indices, axis=1)

    # Step 2: Compute cumulative sums of the sorted softmax scores
    cumulative_probs = jnp.cumsum(sorted_softmax, axis=1)

    # Step 3: Map the cumulative sums back to the original class order
    inv_sorted_indices = jnp.argsort(sorted_indices, axis=1)
    cumulative_probs_original = jnp.take_along_axis(cumulative_probs, inv_sorted_indices, axis=1)

    # Step 4: Compute the APS conformity scores for all classes
    aps_scores = cumulative_probs_original - softmax_scores

    return aps_scores

def get_RAPS_scores_all(softmax_scores, lambda_param, k_reg):
    n_samples, n_classes = softmax_scores.shape

    # Step 1: Sort the softmax scores in descending order for each sample
    sorted_indices = jnp.argsort(-softmax_scores, axis=1)
    sorted_softmax = jnp.take_along_axis(softmax_scores, sorted_indices, axis=1)

    # Step 2: Compute cumulative sums of the sorted softmax scores
    cumulative_probs = jnp.cumsum(sorted_softmax, axis=1)

    # Step 3: Map the cumulative sums back to the original class order
    inv_sorted_indices = jnp.argsort(sorted_indices, axis=1)
    cumulative_probs_original = jnp.take_along_axis(cumulative_probs, inv_sorted_indices, axis=1)

    # Step 4: Compute the rank of each class (1-based rank)
    ranks = inv_sorted_indices + 1  # Ranks start from 1

    # Step 5: Compute the regularization term
    reg_term = jnp.maximum(lambda_param * (ranks - k_reg), 0)

    # Step 6: Add the regularization term to the cumulative probabilities
    scores = cumulative_probs_original + reg_term

    # Step 7: Compute RAPS scores
    raps_scores = scores - softmax_scores

    return raps_scores



