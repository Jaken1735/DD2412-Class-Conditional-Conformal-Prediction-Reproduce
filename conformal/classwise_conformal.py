### Classwise conformal prediction

import numpy as np
from conformal.metrics import *
from conformal.utils import *
from conformal.standard_conformal import *


def calculate_classwise_q_hat(scores_all, true_labels, n_classes, alpha):
    """Computation of class-specific q-hats"""

    def_q_hat = np.inf

    classwise_q_hats = np.zeros((n_classes,))

    for i in range(n_classes):
        pos = (true_labels == i) # select data where i is the true class
        scores = scores_all[pos]
        classwise_q_hats[i] = compute_conformal_threshold(scores=scores,
                                                          alpha=alpha,
                                                          default_value=def_q_hat)
    return classwise_q_hats


def classwise_pred_sets(q_hats, scores):

    scores = np.array(scores)
    prediction_set = []

    for i in range(len(scores)):
        prediction_set.append(np.where(scores[i,:] <= q_hats)[0])    
    
    return prediction_set


def run_classwise(calibration_scores, 
                  calibration_labels, 
                  validation_scores, 
                  alpha,
                  n_classes=100):
    
    q_hats = calculate_classwise_q_hat(calibration_scores, calibration_labels, n_classes, alpha)

    prediction_set = classwise_pred_sets(q_hats, validation_scores)

    return q_hats, prediction_set



