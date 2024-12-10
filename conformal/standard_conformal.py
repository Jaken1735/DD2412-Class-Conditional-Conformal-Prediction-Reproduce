import jax.numpy as jnp

def computeThreshold(scores, alpha, defaultValue=jnp.inf):
    """
    Computes a threshold based on the scores and a significance level alpha.
    """
    nSamples = scores.shape[0]
    nSamples = jnp.array(nSamples, dtype=jnp.int32)

    if nSamples == 0:
        return float(defaultValue)

    quantileLevel = (nSamples + 1) * (1 - alpha) / nSamples
    if quantileLevel > 1:
        print('Quantile Level exceeded.')
        return float(defaultValue)

    sortedScores = jnp.sort(scores)
    index = jnp.clip(jnp.ceil(quantileLevel * nSamples) - 1, 0, nSamples - 1).astype(jnp.int32)
    return sortedScores[index].item()


def getConformalThreshold(scoresAll, trueLabels, alpha):
    """
    Extracts the true scores and computes the conformal threshold (qHat).
    """
    if scoresAll.ndim == 2:
        # Ensure indices are int32
        indices = jnp.arange(len(trueLabels), dtype=jnp.int32)
        trueLabels = trueLabels.astype(jnp.int32)
        trueScores = scoresAll[indices, trueLabels]
    else:
        trueScores = scoresAll

    return computeThreshold(trueScores, alpha)


def createPredictionSets(scoresAll, threshold):
    """
    Generates prediction sets for each instance based on the threshold.
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold should be a scalar numeric value.")

    predictionSets = []
    for scores in scoresAll:
        indices = jnp.where(scores <= threshold)[0]
        predictionSets.append(indices)

    return predictionSets


def performConformalPrediction(calScoresAll, calLabels, valScoresAll, alpha):
    """
    Performs standard conformal prediction.
    """
    threshold = getConformalThreshold(calScoresAll, calLabels, alpha)
    predictions = createPredictionSets(valScoresAll, threshold)
    return threshold, predictions








