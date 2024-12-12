import numpy as np

def compute_class_specific_coverage(true_labels, set_preds):
    num_classes = max(true_labels) + 1
    class_specific_cov = np.zeros((num_classes,))
    for k in range(num_classes):
        idx = np.where(true_labels == k)[0]
        selected_preds = [set_preds[i] for i in idx]
        num_correct = np.sum([1 if np.any(pred_set == k) else 0 for pred_set in selected_preds])
        class_specific_cov[k] = num_correct / len(selected_preds)
        
    return class_specific_cov

def compute_coverage(true_labels, set_preds):
    true_labels = np.array(true_labels) # Convert to numpy to avoid weird pytorch tensor issues
    num_correct = 0
    for true_label, preds in zip(true_labels, set_preds):
        if true_label in preds:
            num_correct += 1
    set_pred_acc = num_correct / len(true_labels)
    
    return set_pred_acc


def compute_coverage_metrics(true_labels, prediction_sets, alpha):

    num_samples = len(true_labels)
    assert num_samples == len(prediction_sets), "Number of labels and prediction sets must match."

    # Determine if the true label is in the prediction set for each sample
    hits = [true_labels[i] in prediction_sets[i] for i in range(num_samples)]

    # Compute overall coverage
    coverage = np.mean(hits)

    # Compute miscoverage
    miscoverage = 1 - coverage

    # Prepare results
    coverage_results = {
        'coverage': coverage,
        'covGap': coverage - (1 - alpha),
        'miscoverage': miscoverage,
        'expected_coverage': 1 - alpha,
        'expected_miscoverage': alpha,
        'num_samples': num_samples
    }

    return coverage_results


def compute_set_size_metrics(prediction_sets):
    
    set_sizes = [len(pred_set) for pred_set in prediction_sets]

    size_metrics = {
        'mean_size': np.mean(set_sizes),
        'median_size': np.median(set_sizes),
        'max_size': np.max(set_sizes),
        'min_size': np.min(set_sizes),
        'std_size': np.std(set_sizes),
    }

    return size_metrics


def compute_all_metrics(val_labels, preds, alpha, cluster_assignments=None):
    class_cond_cov = compute_class_specific_coverage(val_labels, preds)
        
    # Average class coverage gap
    avg_class_cov_gap = np.mean(np.abs(class_cond_cov - (1-alpha)))

    # Average class coverage gap error
    avg_class_cov_gap_std = np.std(np.abs(class_cond_cov - (1-alpha)))

    # Average gap for classes that are over-covered
    overcov_idx = (class_cond_cov > (1-alpha))
    overcov_gap = np.mean(class_cond_cov[overcov_idx] - (1-alpha))

    # Average gap for classes that are under-covered
    undercov_idx = (class_cond_cov < (1-alpha))
    undercov_gap = np.mean(np.abs(class_cond_cov[undercov_idx] - (1-alpha)))
    
    # Fraction of classes that are at least 10% under-covered
    thresh = .1
    very_undercovered = np.mean(class_cond_cov < (1-alpha-thresh))
    
    # Max gap
    max_gap = np.max(np.abs(class_cond_cov - (1-alpha)))

    # Marginal coverage
    marginal_cov = compute_coverage(val_labels, preds)

    class_cov_metrics = {'mean_class_cov_gap': avg_class_cov_gap, 
                         'cov_gap_std': avg_class_cov_gap_std,
                         'undercov_gap': undercov_gap, 
                         'overcov_gap': overcov_gap, 
                         'max_gap': max_gap,
                         'very_undercovered': very_undercovered,
                         'marginal_cov': marginal_cov,
                         'raw_class_coverages': class_cond_cov,
                         'cluster_assignments': cluster_assignments # Also save class cluster assignments
                        }

    curr_set_sizes = [len(x) for x in preds]
    set_size_metrics = {'set_size_mean': np.mean(curr_set_sizes), 
                        'set_size_std': np.std(curr_set_sizes),
                        '[.25, .5, .75, .9] quantiles': np.quantile(curr_set_sizes, [.25, .5, .75, .9])
                        }
    
    return class_cov_metrics, set_size_metrics