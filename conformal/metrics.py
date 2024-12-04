import numpy as np

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
