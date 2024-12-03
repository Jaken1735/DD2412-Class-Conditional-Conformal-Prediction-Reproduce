import os
import numpy as np

##### EXPERIMENT WORKFLOW #######

# 0: We call the experiment-funcction and pass the following arguments#

"""
* dataset: Name of the dataset to use (e.g., 'cifar-100').
* save_folder: Directory where results will be saved.
* alpha: Miscoverage level (e.g., 0.05 for 95% coverage).
* n_totalcal: Total number of calibration samples (average per class).
* score_function_list: List of score functions to use (e.g., ['softmax', 'APS']).
* methods: List of conformal methods to apply (e.g., ['standard', 'classwise']).
* seeds: List of random seeds for reproducibility.
* cluster_args: Dictionary containing clustering parameters (if using clustered conformal).
* calibration_sampling: Method for splitting data (e.g., 'random', 'balanced').
"""

# 1: Loading Data from the pre-trained model (Softmax Scores and True Labels) #
def load_dataset(dataset_name, data_folder='data'):
    data_path = os.path.join(data_folder, f'{dataset_name}.npz')
    data = np.load(data_path)
    softmax_scores = data['softmax']
    labels = data['labels']
    return softmax_scores, labels


# 2: Computing Conformal Scores (e.g., softmax, APS, RASP)
def compute_softmax_conformity_scores(probabilities):
    """
    Computes conformity scores from softmax probabilities.

    Parameters:
    - probabilities: ndarray of shape (n_samples, n_classes), softmax output probabilities.

    Returns:
    - conformity_scores: ndarray of the same shape, conformity scores.
    """
    return 1 - probabilities  # Lower probabilities indicate higher conformity scores

## ETC for the others if on the list of score functions to look at

# 3: Data Splitting into Calibration and Validation sets based on Calibration Sampling Parameter
"""
Here we want to call the split data function, and choose between random and balanced split based on the parameter
"""

def random_split(X, y, avg_num_per_class, seed=0):
    np.random.seed(seed)
    num_classes = np.max(y) + 1
    num_samples = avg_num_per_class * num_classes
    idx1 = np.random.choice(np.arange(len(y)), size=num_samples, replace=False)
    idx2 = ~np.isin(np.arange(len(y)), idx1)
    X1, y1 = X[idx1], y[idx1]
    X2, y2 = X[idx2], y[idx2]
    return X1, y1, X2, y2


def split_X_and_y(X, y, n_k, num_classes, seed=0, split='balanced'):
    np.random.seed(seed)
    if split == 'balanced':
        n_k = n_k * np.ones((num_classes,), dtype=int)
    elif split == 'proportional':
        # Adjust n_k based on class frequencies
        pass
    else:
        raise Exception('Valid split options are "balanced" or "proportional"')
    X1, y1 = [], []
    all_selected_indices = np.zeros(y.shape)
    for k in range(num_classes):
        idx = np.argwhere(y == k).flatten()
        selected_idx = np.random.choice(idx, replace=False, size=(n_k[k],))
        X1.append(X[selected_idx])
        y1.append(y[selected_idx])
        all_selected_indices[selected_idx] = 1
    X1 = np.concatenate(X1)
    y1 = np.concatenate(y1)
    X2 = X[all_selected_indices == 0]
    y2 = y[all_selected_indices == 0]
    return X1, y1, X2, y2


# 4: Applying Conformal Mehtods

###### STANDARD CONFORMAL PIPELINE #######
# Standard conformal pipeline
def standard_conformal(cal_scores_all, cal_labels, val_scores_all, val_labels, alpha, exact_coverage=False):
    '''
    Use cal_scores_all and cal_labels to compute 1-alpha conformal quantiles for standard conformal.
    If exact_coverage is True, apply randomized to achieve exact 1-alpha coverage. Otherwise, use
    unrandomized conservative sets. 
    Create predictions and compute evaluation metrics on val_scores_all and val_labels.
    '''
    
    standard_qhat = compute_qhat(cal_scores_all, cal_labels, alpha, exact_coverage=exact_coverage)
    standard_preds = create_prediction_sets(val_scores_all, standard_qhat, exact_coverage=exact_coverage)
    coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, standard_preds, alpha)
    
    return standard_qhat, standard_preds, coverage_metrics, set_size_metrics


###### CLASSWISE CONFORMAL PIPELINE #######

# Classwise conformal pipeline
def classwise_conformal(totalcal_scores_all, totalcal_labels, val_scores_all, val_labels, alpha,
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    '''
    Use cal_scores_all and cal_labels to compute 1-alpha conformal quantiles for classwise conformal.
    If exact_coverage is True, apply randomized to achieve exact 1-alpha coverage. Otherwise, use
    unrandomized conservative sets. 
    Create predictions and compute evaluation metrics on val_scores_all and val_labels.
    
    See compute_class_specific_qhats() docstring for more details about expected inputs.
    '''
    
    classwise_qhats = compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alpha=alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    classwise_preds = create_classwise_prediction_sets(val_scores_all, classwise_qhats, exact_coverage=exact_coverage)

    coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
    
    return classwise_qhats, classwise_preds, coverage_metrics, set_size_metrics
    

###### CLUSTERED CONFORMAL PIPELINE #######

def clustered_conformal(totalcal_scores_all, totalcal_labels,
                        alpha,
                        val_scores_all=None, val_labels=None,
                        frac_clustering='auto', num_clusters='auto',
                        split='random',
                        exact_coverage=False, seed=0):
        
    np.random.seed(seed) 
    
    def get_rare_classes(labels, alpha, num_classes):
        thresh = get_quantile_threshold(alpha)
        classes, cts = np.unique(labels, return_counts=True)
        rare_classes = classes[cts < thresh]
        
        # Also included any classes that are so rare that we have 0 labels for it
        zero_ct_classes = np.setdiff1d(np.arange(num_classes), classes)
        rare_classes = np.concatenate((rare_classes, zero_ct_classes))
        
        return rare_classes
        
    def remap_classes(labels, rare_classes):
        '''
        Exclude classes in rare_classes and remap remaining classes to be 0-indexed

        Outputs:
            - remaining_idx: Boolean array the same length as labels. Entry i is True
            iff labels[i] is not in rare_classes 
            - remapped_labels: Array that only contains the entries of labels that are 
            not in rare_classes (in order) 
            - remapping: Dict mapping old class index to new class index

        '''
        remaining_idx = ~np.isin(labels, rare_classes)

        remaining_labels = labels[remaining_idx]
        remapped_labels = np.zeros(remaining_labels.shape, dtype=int)
        new_idx = 0
        remapping = {}
        for i in range(len(remaining_labels)):
            if remaining_labels[i] in remapping:
                remapped_labels[i] = remapping[remaining_labels[i]]
            else:
                remapped_labels[i] = new_idx
                remapping[remaining_labels[i]] = new_idx
                new_idx += 1
        return remaining_idx, remapped_labels, remapping
    
    # Data preperation: Get conformal scores for true classes
    num_classes = totalcal_scores_all.shape[1]
    totalcal_scores = get_true_class_conformal_score(totalcal_scores_all, totalcal_labels)
    
    # 1) Apply heuristic to choose hyperparameters if not prespecified
    if frac_clustering == 'auto' and num_clusters == 'auto':
        cts_dict = Counter(totalcal_labels)
        cts = [cts_dict.get(k, 0) for k in range(num_classes)]
        n_min = min(cts)
        n_thresh = get_quantile_threshold(alpha) 
        n_min = max(n_min, n_thresh) # Classes with fewer than n_thresh examples will be excluded from clustering
        num_remaining_classes = np.sum(np.array(list(cts)) >= n_min)

        n_clustering, num_clusters = get_clustering_parameters(num_remaining_classes, n_min)
        print(f'n_clustering={n_clustering}, num_clusters={num_clusters}')
        # Convert n_clustering to fraction relative to n_min
        frac_clustering = n_clustering / n_min

        
    # 2a) Split data
    if split == 'proportional':
        n_k = [int(frac_clustering*cts[k]) for k in range(num_classes)]
        scores1, labels1, scores2, labels2 = split_X_and_y(totalcal_scores, 
                                                           totalcal_labels, 
                                                           n_k, 
                                                           num_classes=num_classes, 
                                                           seed=0)
#                                                            split=split, # Balanced or stratified sampling 
    elif split == 'doubledip':
        scores1, labels1 = totalcal_scores, totalcal_labels
        scores2, labels2 = totalcal_scores, totalcal_labels
    elif split == 'random':
        # Each point is assigned to clustering set w.p. frac_clustering 
        idx1 = np.random.uniform(size=(len(totalcal_labels),)) < frac_clustering 
        scores1 = totalcal_scores[idx1]
        labels1 = totalcal_labels[idx1]
        scores2 = totalcal_scores[~idx1]
        labels2 = totalcal_labels[~idx1]
        
    else:
        raise Exception('Invalid split. Options are balanced, proportional, doubledip, and random')

    # 2b)  Identify "rare" classes = classes that have fewer than 1/alpha - 1 examples 
    # in the clustering set 
    rare_classes = get_rare_classes(labels1, alpha, num_classes)
    print(f'{len(rare_classes)} of {num_classes} classes are rare in the clustering set'
          ' and will be assigned to the null cluster')
    
    # 3) Run clustering
    if num_classes - len(rare_classes) > num_clusters and num_clusters > 1:  
        # Filter out rare classes and re-index
        remaining_idx, filtered_labels, class_remapping = remap_classes(labels1, rare_classes)
        filtered_scores = scores1[remaining_idx]
        
        # Compute embedding for each class and get class counts
        embeddings, class_cts = embed_all_classes(filtered_scores, filtered_labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=True)
    
        kmeans = KMeans(n_clusters=int(num_clusters), random_state=0, n_init=10).fit(embeddings, sample_weight=np.sqrt(class_cts))
        nonrare_class_cluster_assignments = kmeans.labels_  

        # Print cluster sizes
        print(f'Cluster sizes:', [x[1] for x in Counter(nonrare_class_cluster_assignments).most_common()])

        # Remap cluster assignments to original classes. Any class not included in kmeans clustering is a rare 
        # class, so we will assign it to cluster "-1" = num_clusters by Python indexing
        cluster_assignments = -np.ones((num_classes,), dtype=int)
        for cls, remapped_cls in class_remapping.items():
            cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
    else: 
        cluster_assignments = -np.ones((num_classes,), dtype=int)
        print('Skipped clustering because the number of clusters requested was <= 1')
        
    # 4) Compute qhats for each cluster
    cal_scores_all = scores2
    cal_labels = labels2
    if exact_coverage: 
        q_as, q_bs, gammas = compute_cluster_specific_qhats(cluster_assignments, cal_scores_all, cal_labels, alpha, 
                                   null_qhat='standard', exact_coverage=True)
        
    else: 
        qhats = compute_cluster_specific_qhats(cluster_assignments, 
                   cal_scores_all, cal_labels, 
                   alpha=alpha, 
                   null_qhat='standard')
        

    # 5) [Optionally] Apply to val set. Evaluate class coverage gap and set size 
    if (val_scores_all is not None) and (val_labels is not None):
        if exact_coverage:
            preds = construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, val_scores_all)
            qhats = {'q_a': q_as, 'q_b': q_bs, 'gamma': gammas} # Alias for function return
        else:
            preds = create_classwise_prediction_sets(val_scores_all, qhats)
        class_cov_metrics, set_size_metrics = compute_all_metrics(val_labels, preds, alpha,
                                                                  cluster_assignments=cluster_assignments)

        # Add # of classes excluded from clustering to class_cov_metrics
        class_cov_metrics['num_unclustered_classes'] = len(rare_classes)
        
        return qhats, preds, class_cov_metrics, set_size_metrics
    else:
        return qhats