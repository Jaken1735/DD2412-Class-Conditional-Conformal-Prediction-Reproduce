import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Verify sys.path
print("PYTHONPATH:", sys.path)


import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans

# Ensure necessary functions are imported or defined
from conformal.utils import random_split, compute_softmax_conformity_scores
from conformal.metrics import compute_coverage_metrics, compute_set_size_metrics
from conformal.clustered_conformal import embed_all_classes, identify_rare_classes, compute_cluster_quantiles, generate_prediction_sets

# Set random seed
np.random.seed(42)

# Step 1: Load CIFAR-100 Data from .npy files
def load_cifar100_data(scores_file='data/results_scores.npy', labels_file='data/results_labels.npy'):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scores_path = os.path.join(base_dir, scores_file)
    labels_path = os.path.join(base_dir, labels_file)
    softmax_scores = np.load(scores_path)
    labels = np.load(labels_path)
    return softmax_scores, labels

# Load data
softmax_scores, labels = load_cifar100_data()

# Split Data
X_calib, y_calib, X_valid, y_valid = random_split(softmax_scores, labels, avg_num_per_class=50, seed=42)

# Step 3: Compute Conformity Scores
calib_conformity_scores = compute_softmax_conformity_scores(X_calib)
valid_conformity_scores = compute_softmax_conformity_scores(X_valid)

# Step 4: Data Splitting for Clustering
frac_clustering = 0.5
mask = np.random.rand(len(y_calib)) < frac_clustering

# Clustering set
scores_cluster = calib_conformity_scores[mask]
y_cluster = y_calib[mask]

# Calibration set
scores_calib_final = calib_conformity_scores[~mask]
y_calib_final = y_calib[~mask]

# Step 5: Rare Class Identification
min_samples = 10
rare_classes = identify_rare_classes(y_cluster, min_samples)

# Filter out rare classes for clustering
nonrare_mask = ~np.isin(y_cluster, rare_classes)
scores_cluster_nonrare = scores_cluster[nonrare_mask]
y_cluster_nonrare = y_cluster[nonrare_mask]

# Step 6: Compute Class Embeddings
embeddings, class_cts = embed_all_classes(scores_cluster_nonrare, y_cluster_nonrare, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=True)

# Step 7: Clustering
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(embeddings, sample_weight=np.sqrt(class_cts))
cluster_labels = kmeans.labels_

# Map classes to clusters
unique_classes = np.unique(y_cluster_nonrare)
class_to_cluster = {cls: cluster for cls, cluster in zip(unique_classes, cluster_labels)}

# Assign rare classes to null cluster (-1)
for cls in rare_classes:
    class_to_cluster[cls] = -1

# Assign any classes not yet in class_to_cluster to the null cluster (-1)
all_classes_in_calib = np.unique(y_calib_final)
for cls in all_classes_in_calib:
    if cls not in class_to_cluster:
        class_to_cluster[cls] = -1


# Step 8: Compute Cluster-Specific Quantiles
alpha = 0.05
cluster_quantiles = compute_cluster_quantiles(
    scores_calib_final, y_calib_final, class_to_cluster, alpha
)

# Step 9: Assign Validation Samples to Clusters
# Map true labels to clusters (or use predicted labels if appropriate)
valid_clusters = np.array([class_to_cluster.get(cls, -1) for cls in y_valid])

# Step 10: Generate Prediction Sets
predictions = generate_prediction_sets(valid_conformity_scores, valid_clusters, cluster_quantiles)

# Step 11: Evaluate Results
coverage_metrics = compute_coverage_metrics(y_valid, predictions, alpha)
print(f"Coverage Metrics:\n{coverage_metrics}")

set_size_metrics = compute_set_size_metrics(predictions)
print(f"Set Size Metrics:\n{set_size_metrics}")
