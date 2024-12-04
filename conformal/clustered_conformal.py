import numpy as np

def identify_rare_classes(labels, min_samples):
    """
    Identifies classes with fewer than min_samples instances.
    
    Parameters:
    - labels: array-like, class labels.
    - min_samples: int, minimum number of samples required.
    
    Returns:
    - rare_classes: list of class labels considered rare.
    """
    from collections import Counter
    label_counts = Counter(labels)
    rare_classes = [label for label, count in label_counts.items() if count < min_samples]
    return rare_classes


def compute_cluster_quantiles(calib_scores, calib_labels, class_to_cluster, alpha):
    from collections import defaultdict
    
    # Organize scores by cluster
    cluster_scores = defaultdict(list)
    for score, label in zip(calib_scores, calib_labels):
        cluster_id = class_to_cluster[label]
        cluster_scores[cluster_id].append(score)
    
    # Compute standard conformal quantile as default
    n_total = len(calib_scores)
    if n_total == 0:
        standard_q_hat = np.inf
    else:
        val = np.ceil((n_total + 1) * (1 - alpha)) / n_total
        if val > 1:
            standard_q_hat = np.inf
        else:
            standard_q_hat = np.quantile(calib_scores, val, method='higher')
    
    # Compute quantiles for each cluster
    cluster_quantiles = {}
    for cluster_id, scores in cluster_scores.items():
        scores = np.array(scores)
        n = len(scores)
        if n == 0:
            q_hat = standard_q_hat  # Use standard conformal quantile
        else:
            quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
            if quantile_level > 1:
                q_hat = standard_q_hat
            else:
                q_hat = np.quantile(scores, quantile_level, method='higher')
        cluster_quantiles[cluster_id] = q_hat
    
    return cluster_quantiles



def generate_prediction_sets(valid_scores, valid_clusters, cluster_quantiles):
    """
    Generates prediction sets for validation data using cluster-specific quantiles.
    
    Parameters:
    - valid_scores: array-like, conformity scores for validation data.
    - valid_clusters: array-like, cluster IDs for validation data.
    - cluster_quantiles: dict, mapping cluster IDs to quantiles.
    
    Returns:
    - prediction_sets: list of arrays, prediction sets for each validation sample.
    """
    prediction_sets = []
    for scores, cluster_id in zip(valid_scores, valid_clusters):
        q_hat = cluster_quantiles.get(cluster_id, np.inf)  # Use a default value if cluster not found
        pred_set = np.where(scores <= q_hat)[0]
        prediction_sets.append(pred_set)
    return prediction_sets


import numpy as np

def embed_all_classes(scores, labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=False):
    """
    Computes embeddings for each class based on specified quantiles of their scores.

    Parameters:
    - scores: array-like, conformity scores for the samples (e.g., scores for the true class)
              Shape: (num_samples,)
    - labels: array-like, class labels corresponding to the scores
              Shape: (num_samples,)
    - q: list of quantiles to compute for each class
         Default: [0.5, 0.6, 0.7, 0.8, 0.9]
    - return_cts: bool, whether to return class counts along with embeddings

    Returns:
    - embeddings: np.array of shape (num_classes, len(q)), where each row corresponds to a class's embedding
    - class_cts (optional): array of class counts for each class, shape: (num_classes,)
    """
    # Get the unique classes and their indices
    classes = np.unique(labels)
    num_classes = len(classes)

    # Map from class label to index
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    # Initialize embeddings array
    embeddings = np.zeros((num_classes, len(q)))
    class_cts = np.zeros(num_classes, dtype=int)

    # For each class, compute the quantiles of the scores
    for cls in classes:
        idx = (labels == cls)
        cls_scores = scores[idx]
        class_idx = class_to_index[cls]

        if len(cls_scores) == 0:
            # If there are no scores for this class, fill with zeros or NaNs
            embeddings[class_idx, :] = np.nan  # or use zeros if preferred
            class_cts[class_idx] = 0
        else:
            quantiles = np.quantile(cls_scores, q)
            embeddings[class_idx, :] = quantiles
            class_cts[class_idx] = len(cls_scores)

    if return_cts:
        return embeddings, class_cts
    else:
        return embeddings





