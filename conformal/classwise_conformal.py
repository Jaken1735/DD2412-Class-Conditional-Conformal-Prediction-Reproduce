import jax.numpy as jnp
from conformal.metrics import *
from conformal.utils import *
from conformal.standard_conformal import *


def calculate_classwise_q_hat(scores_all, true_labels, n_classes, alpha):
    """Computation of class-specific q-hats"""
    if len(scores_all.shape) == 2:
        scores_all = scores_all[jnp.arange(len(true_labels)), true_labels]

    classwise_q_hats = jnp.zeros((n_classes,))

    def compute_class_q_hat(class_idx):
        pos = (true_labels == class_idx)
        scores = scores_all[pos]
        return computeThreshold(scores=scores, alpha=alpha, defaultValue=jnp.inf)

    classwise_q_hats = jnp.array([compute_class_q_hat(i) for i in range(n_classes)])
    
    return classwise_q_hats


def classwise_pred_sets(q_hats, scores):

    scores = jnp.array(scores)
    prediction_set = []

    for i in range(len(scores)):
        prediction_set.append(jnp.where(scores[i, :] <= q_hats)[0])    
    
    return prediction_set


def run_classwise(calibration_scores, 
                  calibration_labels, 
                  validation_scores, 
                  alpha,
                  n_classes=100):
    
    q_hats = calculate_classwise_q_hat(calibration_scores, calibration_labels, n_classes, alpha)
    prediction_set = classwise_pred_sets(q_hats, validation_scores)

    return q_hats, prediction_set




