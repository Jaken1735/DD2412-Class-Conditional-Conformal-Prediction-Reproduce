import numpy as np
from collections import Counter

def quantileThreshold(alpha):
    '''
    Compute the smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
    '''
    n = 1
    while np.ceil((n + 1) * (1 - alpha) / n) > 1:
        n += 1
    return n

def rareClasses(labels, alpha, num_classes):
    """
    Function which fetches the rare classes based on the set alpha for 
    the quantile threshold
    """
    # Start by computing the quantile threshold
    smallest_n = quantileThreshold(alpha)
    # Fetch number of samples per class, aka get the distribution of all the classes in the dataset
    classes, cts = np.unique(labels, return_counts=True)
    # Determine rare classes based on threshold
    rare_classes = classes[cts < smallest_n]
    # Identify classes with zero samples
    class_set = set(classes)
    zeroSample_classes = [k for k in range(num_classes) if k not in class_set]
    # Merge sets
    rare_classes = np.concatenate((rare_classes, zeroSample_classes))

    return rare_classes


def get_clustering_parameters(num_classes, n_totalcal):
    '''
    Returns a guess of good values for num_clusters and n_clustering based solely 
    on the number of classes and the number of examples per class. 

    This relies on two heuristics:
    1) We want at least 150 points per cluster on average.
    2) We need more samples as we try to distinguish between more distributions. 
       - To distinguish between 2 distributions, want at least 4 samples per class. 
       - To distinguish between 5 distributions, want at least 10 samples per class.

    Output:
    - n_clustering: Number of samples per class to use for clustering.
    - num_clusters: Number of clusters to form.
    '''
    # Alias for convenience
    K = num_classes
    N = n_totalcal

    n_clustering = int(N * K / (75 + K))
    num_clusters = int(np.floor(n_clustering / 2))

    return n_clustering, num_clusters

def selecting_hparameters(totalcal_labels, num_classes, alpha):
    """
    Function for selecting optimal hyperparameters for clustering.

    Inputs:
    - totalcal_labels: array of labels in the total calibration dataset.
    - num_classes: total number of classes.
    - alpha: desired miscoverage rate (e.g., 0.1 for 90% coverage).

    Outputs:
    - n_clustering: Number of samples per class to use for clustering.
    - num_clusters: Number of clusters to form.
    - frac_clustering: Fraction of data to use for clustering.
    """
    #np.random.seed(0)
    # Step 1: Count the number of samples per class
    cts_dict = Counter(totalcal_labels)
    cts = [cts_dict.get(k, 0) for k in range(num_classes)]
    
    # Step 2: Compute n_min
    n_min = min(cts)
    
    # Step 3: Compute n_thresh
    n_thresh = quantileThreshold(alpha)
    
    # Step 4: Adjust n_min
    n_min = max(n_min, n_thresh)  # Classes with fewer than n_thresh examples will be excluded from clustering
    
    # Step 5: Determine the number of classes with sufficient samples
    num_remaining_classes = np.sum(np.array(cts) >= n_min)
    
    # Step 6: Compute clustering parameters
    n_clustering, num_clusters = get_clustering_parameters(num_remaining_classes, n_min)
    print(f'n_clustering={n_clustering}, num_clusters={num_clusters}')
    
    # Step 7: Compute frac_clustering
    frac_clustering = n_clustering / n_min
    
    return n_clustering, num_clusters, frac_clustering



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


def embed_all_classes(scores_all, labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=False):
    '''
    Inputs:
        - scores_all: Either
            - A (num_instances x num_classes) array where scores_all[i, j] is the score for class j of instance i.
            - A (num_instances,) array where scores_all[i] is the score for the true class of instance i.
        - labels: A (num_instances,) array of true class labels for each instance.
        - q: A list of quantiles to compute for each class.
        - return_cts: If True, also return the count of samples per class.

    Outputs:
        - embeddings: A (num_classes x len(q)) array where each row corresponds to the quantile embeddings of a class.
        - (Optional) cts: A (num_classes,) array where cts[i] is the number of samples for class i.
    '''
    num_classes = np.max(labels) + 1  # Assuming classes are 0-indexed and contiguous

    if scores_all.ndim == 2:
        # Get the score of the true class for each instance
        scores = scores_all[np.arange(labels.shape[0]), labels]
    else:
        scores = scores_all

    # Initialize arrays
    embeddings = np.full((num_classes, len(q)), np.nan)
    cts = np.zeros(num_classes, dtype=int)

    # Loop over present classes only
    unique_labels = np.unique(labels)
    for cls in unique_labels:
        class_scores = scores[labels == cls]
        cts[cls] = class_scores.shape[0]
        if cts[cls] > 0:
            embeddings[cls, :] = np.quantile(class_scores, q)
        else:
            embeddings[cls, :] = np.nan  # Handle classes with no samples

    if return_cts:
        return embeddings, cts
    else:
        return embeddings
    

def compute_qhat(scores, labels, alpha):
    """
    Computes the quantile q_hat for standard conformal prediction.

    Inputs:
        - scores: NumPy array of conformity scores for the true class.
        - labels: NumPy array of true class labels (not used in this function).
        - alpha: Desired miscoverage rate (e.g., 0.1 for 90% coverage).

    Output:
        - q_hat: The quantile used for conformal prediction.
    """
    n = len(scores)
    # Compute the index k
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)  # Ensure k is within bounds
    # Sort the scores
    sorted_scores = np.sort(scores)
    # Get q_hat
    q_hat = sorted_scores[k - 1]  # Adjust for zero-based indexing
    return q_hat


def compute_class_specific_qhats(scores, clusters, alpha, num_classes, default_qhat=np.inf, null_qhat=None):
    """
    Computes class-specific quantiles (q_hats) for each class or cluster.

    Inputs:
        - scores: NumPy array of conformity scores for the true class.
        - clusters: NumPy array of cluster assignments for each instance.
        - alpha: Desired miscoverage rate.
        - num_classes: Total number of clusters/classes.
        - default_qhat: Default quantile value for clusters with insufficient data.
        - null_qhat: Quantile value to use for clusters with no data.

    Outputs:
        - qhats: NumPy array of q_hat values per cluster.
    """
    # Initialize qhats with default values
    qhats = np.full(num_classes, default_qhat)
    for c in range(num_classes):
        # Get scores for cluster c
        cluster_scores = scores[clusters == c]
        n = len(cluster_scores)
        if n == 0:
            # No data for this cluster
            if null_qhat is not None:
                qhats[c] = null_qhat
            else:
                qhats[c] = default_qhat
            continue
        # Compute q_hat for cluster c
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = min(max(k, 1), n)
        sorted_scores = np.sort(cluster_scores)
        qhats[c] = sorted_scores[k - 1]
    return qhats


def compute_cluster_specific_qhats(cluster_assignments, cal_scores_all, cal_true_labels, alpha, 
                                   null_qhat='standard', exact_coverage=False):
    if null_qhat == 'standard' and not exact_coverage:
        null_qhat = compute_qhat(cal_scores_all, cal_true_labels, alpha)
        
    if cal_scores_all.ndim == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]
        
    if np.all(cluster_assignments == -1):
        if exact_coverage:
            null_qa, null_qb, null_gamma = get_exact_coverage_conformal_params(cal_scores_all, alpha)
            q_as = np.full(cluster_assignments.shape, null_qa)
            q_bs = np.full(cluster_assignments.shape, null_qb)
            gammas = np.full(cluster_assignments.shape, null_gamma)
            return q_as, q_bs, gammas
        else:
            return np.full(cluster_assignments.shape, null_qhat)
    
    # Map true class labels to clusters
    cal_true_clusters = cluster_assignments[cal_true_labels]
    
    cluster_qhats = compute_class_specific_qhats(
        cal_scores_all, cal_true_clusters, 
        alpha=alpha, 
        num_classes=np.max(cluster_assignments) + 1,
        default_qhat=np.inf,
        null_qhat=null_qhat
    )
        
    # Map cluster qhats back to classes
    num_classes = len(cluster_assignments)
    class_qhats = np.full(num_classes, null_qhat)
        
    valid_clusters = cluster_assignments >= 0
    class_qhats[valid_clusters] = cluster_qhats[cluster_assignments[valid_clusters]]
        
    return class_qhats


    








