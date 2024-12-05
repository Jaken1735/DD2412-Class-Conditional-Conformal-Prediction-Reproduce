import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

print("PYTHONPATH:", sys.path)

import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Ensure necessary functions are imported or defined
from conformal.utils import random_split, compute_softmax_conformity_scores, reinitClasses
from conformal.metrics import create_classwise_prediction_sets, compute_all_metrics
from conformal.clustered_conformal import embed_all_classes, rareClasses, compute_cluster_specific_qhats, generate_prediction_sets, selecting_hparameters


def load_cifar100_data(scores_file='data/results_scores.npy', labels_file='data/results_labels.npy'):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scores_path = os.path.join(base_dir, scores_file)
    labels_path = os.path.join(base_dir, labels_file)
    softmax_scores = np.load(scores_path)
    labels = np.load(labels_path)
    return softmax_scores, labels

softmax_scores, labels = load_cifar100_data()
#print(softmax_scores[0])
#print(labels[0])
print('Predicted Index: ', np.max(labels[0]))

#### PARAMETERS ####
SEED = 0
np.random.seed(SEED)
N_AVG = 60
###################

# Compute Conformal Score
conformal_scores_all = 1 - softmax_scores # Using Softmax

# Randomly Split Data
# Adjust avg_num_per_class as needed
totalcal_scores, totalcal_labels, val_scores, val_labels = random_split(conformal_scores_all, labels, avg_num_per_class=N_AVG, seed=SEED)

# Choosing hparameters for clustering
num_classes = 100 # CIFAR-100 Dataset
alpha = 0.1

# Call the function
n_clustering, num_clusters, frac_clustering = selecting_hparameters(totalcal_labels, num_classes, alpha)

print(f"Number of samples per class for clustering: {n_clustering}")
print(f"Number of clusters: {num_clusters}")
print(f"Fraction of data per class to use for clustering: {frac_clustering}")

# Each point is assigned to clustering set w.p. frac_clustering
idx1 = np.random.uniform(size=(len(totalcal_labels),)) < frac_clustering 
scores1 = totalcal_scores[idx1]
labels1 = totalcal_labels[idx1]
scores2 = totalcal_scores[~idx1]
labels2 = totalcal_labels[~idx1]

# Now we want to identify the rare classes based on the set alpha
rare_classes = rareClasses(labels=labels1, alpha=alpha, num_classes=num_classes)
print(f'{len(rare_classes)} of {num_classes} classes are rare in the clustering set and will be assigned to the null cluster')

# Start Clustering of Non-rare classes in order to compute cluster-specific conformal quantiles
num_nonrare_classes = num_classes - len(rare_classes)
if num_nonrare_classes >= 2 and num_clusters >= 2: # Check if we can start clustering
    # Filter out rare classes and re-index
    remaining_idx, filtered_labels, class_remapping = reinitClasses(labels1, rare_classes)
    filtered_scores = scores1[remaining_idx]
    # Compute embedding for each class and get class counts
    embeddings, class_cts = embed_all_classes(
        filtered_scores, filtered_labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=True
    )

    """
    KMEANS Clustering Model
    """
    kmeans = KMeans(n_clusters=int(num_clusters), random_state=0, n_init=10).fit(
        embeddings, sample_weight=np.sqrt(class_cts)
    )
    nonrare_class_cluster_assignments = kmeans.labels_


    """
    Gaussian Mixture Model
    """
    #gmm = GaussianMixture(n_components=int(num_clusters), random_state=0)
    #gmm.fit(embeddings)
    #nonrare_class_cluster_assignments = gmm.predict(embeddings)

    """
    Agglomerative Clustering
    """
    #agglo = AgglomerativeClustering(n_clusters=int(num_clusters))
    #onrare_class_cluster_assignments = agglo.fit_predict(embeddings)

    # Print cluster sizes
    cluster_sizes = [x[1] for x in Counter(nonrare_class_cluster_assignments).most_common()]
    print(f'Cluster sizes:', cluster_sizes)

    # Remap cluster assignments to original classes
    cluster_assignments = -np.ones((num_classes,), dtype=int)
    for cls, remapped_cls in class_remapping.items():
        cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
else:
    cluster_assignments = -np.ones((num_classes,), dtype=int)
    print('Skipped clustering due to insufficient classes or clusters.')


# Now we want to compute qhats for the clussters
cal_scores_all = scores2
cal_labels = labels2
qhats = compute_cluster_specific_qhats(cluster_assignments, cal_scores_all, cal_labels, alpha=alpha, null_qhat='standard')
preds = create_classwise_prediction_sets(val_scores, qhats)
class_cov_metrics, set_size_metrics = compute_all_metrics(val_labels, preds, alpha, cluster_assignments=cluster_assignments)
