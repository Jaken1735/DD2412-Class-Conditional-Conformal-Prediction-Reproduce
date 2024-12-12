import sys
import os
import argparse
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from conformal.utils import random_split, reinitClasses, compute_APS_scores, get_RAPS_scores_all
from conformal.metrics import compute_all_metrics
from conformal.clustered_conformal import embed_all_classes, rareClasses, clusterSpecificQhats, selecting_hparameters
from conformal.classwise_conformal import classwise_pred_sets

def load_cifar100_data(scores_file='data/results_scores.npy', labels_file='data/results_labels.npy'):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scores_path = os.path.join(base_dir, scores_file)
    labels_path = os.path.join(base_dir, labels_file)
    
    softmax_scores = np.load(scores_path, mmap_mode='r')
    labels = np.load(labels_path, mmap_mode='r')
    return softmax_scores, labels

#### PARAMETERS ####
lmbda = 0.0005
kreg = 50
alpha = 0.1
SEED = 2
np.random.seed(SEED)
###################

parser = argparse.ArgumentParser()
parser.add_argument('--N_AVG', type=int, default=10, help='Average number per class for calibration')
parser.add_argument('--score_func', nargs='+', default=['softmax'], help='Example: --score_func softmax APS RAPS')
args = parser.parse_args()

softmax_scores, labels = load_cifar100_data()

for sf in args.score_func:
    if sf == 'softmax':
        conformal_scores_all = 1 - softmax_scores
    elif sf == 'APS':
        conformal_scores_all = compute_APS_scores(softmax_scores)
    elif sf == 'RAPS':
        conformal_scores_all = get_RAPS_scores_all(softmax_scores, lmbda, kreg)
    else:
        raise ValueError(f"Unknown scoring function: {sf}")

    # Randomly Split Data
    totalcal_scores, totalcal_labels, val_scores, val_labels = random_split(
        conformal_scores_all, labels, avg_num_per_class=args.N_AVG
    )

    num_classes = totalcal_scores.shape[1]

    # Choose hyperparameters
    n_clustering, num_clusters, frac_clustering = selecting_hparameters(totalcal_labels, num_classes, alpha)

    # Indexing once for clustering set
    uniform_draws = np.random.uniform(size=(len(totalcal_labels),))
    idx1 = uniform_draws < frac_clustering
    scores1 = totalcal_scores[idx1]
    labels1 = totalcal_labels[idx1]
    scores2 = totalcal_scores[~idx1]
    labels2 = totalcal_labels[~idx1]

    # Identify rare classes
    rare_classes_set = rareClasses(labels1, alpha, num_classes)
    num_rare = len(rare_classes_set)

    num_nonrare_classes = num_classes - num_rare
    if num_nonrare_classes >= 2 and num_clusters >= 2:
        # Reindex classes
        remaining_idx, filtered_labels, class_remapping = reinitClasses(labels1, rare_classes_set)
        filtered_scores = scores1[remaining_idx]

        # Compute embeddings and counts
        embeddings, class_cts = embed_all_classes(
            filtered_scores, filtered_labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=True
        )

        # Clustering (KMeans)
        kmeans = KMeans(n_clusters=int(num_clusters), random_state=0, n_init=10)
        nonrare_class_cluster_assignments = kmeans.fit(embeddings, sample_weight=np.sqrt(class_cts)).labels_
        
        # Report cluster sizes
        cluster_count = Counter(nonrare_class_cluster_assignments)
        cluster_sizes = [count for _, count in cluster_count.most_common()]

        # Remap cluster assignments
        cluster_assignments = -np.ones(num_classes, dtype=int)
        for cls, remapped_cls in class_remapping.items():
            cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
    else:
        # Skip clustering due to insufficient classes or clusters
        cluster_assignments = -np.ones(num_classes, dtype=int)
    
    # Compute qhats
    qhats = clusterSpecificQhats(cluster_assignments, scores2, labels2, alpha)
    preds = classwise_pred_sets(qhats, val_scores)

    # Compute metrics
    coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, preds, alpha, cluster_assignments=cluster_assignments)

    variables = f"clustered,{sf},{args.N_AVG},"
    cov_metrics = f"{coverage_metrics['mean_class_cov_gap']},{coverage_metrics['cov_gap_std']},"
    set_metrics = f"{set_size_metrics['set_size_mean']},{set_size_metrics['set_size_std']}"
    print(variables+cov_metrics+set_metrics)