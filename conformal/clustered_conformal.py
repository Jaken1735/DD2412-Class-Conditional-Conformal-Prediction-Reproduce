import jax.numpy as jnp
from collections import Counter
import numpy as np

def quantileThreshold(significanceLevel):
    """
    Compute the smallest sampleCount such that ceil((sampleCount+1)*(1-significanceLevel)/sampleCount) <= 1
    """
    sampleCount = 1
    while jnp.ceil((sampleCount + 1) * (1 - significanceLevel) / sampleCount) > 1:
        sampleCount += 1
    return sampleCount

def rareClasses(classLabels, significanceLevel, numberOfClasses):
    """
    Determine the rare classes based on the given significanceLevel and numberOfClasses.
    A class is considered rare if it has fewer samples than the smallest threshold determined by quantileThreshold.
    """
    # Compute the smallest sample count threshold
    smallestSampleCount = quantileThreshold(significanceLevel)
    
    # Get the unique classes and their counts
    uniqueClasses, uniqueCounts = jnp.unique(classLabels, return_counts=True)
    
    # Identify classes with counts less than smallestSampleCount
    rareClasses = uniqueClasses[uniqueCounts < smallestSampleCount]
    
    # Identify classes with zero samples
    classSet = set(uniqueClasses.tolist())
    zeroSampleClasses = [cls for cls in range(numberOfClasses) if cls not in classSet]
    
    # Combine rare and zero-sample classes
    rareClasses = jnp.concatenate((rareClasses, jnp.array(zeroSampleClasses)))
    
    return rareClasses

def get_clustering_parameters(numberOfClasses, totalCalibrationSamples):
    """
    Compute the number of samples used for clustering and the number of clusters
    based on the numberOfClasses and totalCalibrationSamples.
    """
    classCount = numberOfClasses
    calibrationCount = totalCalibrationSamples
    
    clusteringSampleCount = int(calibrationCount * classCount / (75 + classCount))
    clusterCount = int(jnp.floor(clusteringSampleCount / 2))
    
    return clusteringSampleCount, clusterCount

def selecting_hparameters(totalCalibrationLabels, numberOfClasses, significanceLevel):
    """
    Select hyperparameters for clustering given the labels, numberOfClasses, and significanceLevel.
    """
    # Count the number of samples per class
    classCountDict = Counter(np.asarray(totalCalibrationLabels))
    classCounts = [classCountDict.get(cls, 0) for cls in range(numberOfClasses)]
    
    minCount = min(classCounts)
    thresholdCount = quantileThreshold(significanceLevel)
    minCount = max(minCount, thresholdCount)
    remainingClassCount = jnp.sum(jnp.array(classCounts) >= minCount)
    
    clusteringSampleCount, clusterCount = get_clustering_parameters(remainingClassCount, minCount)
    print(f'clusteringSampleCount={clusteringSampleCount}, clusterCount={clusterCount}')
    
    # Compute fraction of samples for clustering
    clusteringFraction = clusteringSampleCount / minCount
    
    return clusteringSampleCount, clusterCount, clusteringFraction

from collections import defaultdict

def classSpecificQhats(sampleScores, sampleClusters, significanceLevel, numberOfClasses, defaultQuantile=np.inf, nullQuantile=None):
    """
    Compute class-specific quantiles (q_hats) for each class or cluster more efficiently.
    """
    quantiles = np.full(numberOfClasses, defaultQuantile)

    # Group scores by cluster index
    clusterDict = defaultdict(list)
    for scoreValue, clusterIndex in zip(sampleScores, sampleClusters):
        if clusterIndex >= 0:
            clusterDict[clusterIndex].append(scoreValue)

    for clusterIndex in range(numberOfClasses):
        clusterScores = clusterDict.get(clusterIndex, [])
        clusterSize = len(clusterScores)
        if clusterSize == 0:
            quantiles[clusterIndex] = nullQuantile if nullQuantile is not None else defaultQuantile
            continue

        # Compute index for the quantile
        quantileIndex = int(np.ceil((clusterSize + 1) * (1 - significanceLevel)))
        quantileIndex = min(max(quantileIndex, 1), clusterSize)

        clusterArray = np.array(clusterScores)
        np.partition(clusterArray, quantileIndex - 1)
        quantiles[clusterIndex] = clusterArray[quantileIndex - 1]

    return quantiles

def clusterSpecificQhats(clusterAssignments, calibrationScoresAll, calibrationTrueLabels, significanceLevel, nullQuantile='standard'):
    """
    Compute quantiles (q_hats) for each class based on cluster assignments.
    """
    if nullQuantile == 'standard':
        nullQuantile = computeQhat(calibrationScoresAll, significanceLevel)
    
    if calibrationScoresAll.ndim == 2:
        calibrationScoresAll = calibrationScoresAll[np.arange(len(calibrationTrueLabels)), calibrationTrueLabels]
    
    # If no clustering was done
    if np.all(clusterAssignments == -1):
        return np.full(clusterAssignments.shape, nullQuantile)

    calibrationTrueClusters = clusterAssignments[calibrationTrueLabels]
    maxClusterIndex = np.max(clusterAssignments)
    numberOfClusters = maxClusterIndex + 1 if maxClusterIndex >= 0 else 0
    
    clusterQuantiles = classSpecificQhats(
        calibrationScoresAll,
        calibrationTrueClusters,
        significanceLevel=significanceLevel,
        numberOfClasses=numberOfClusters,
        defaultQuantile=np.inf,
        nullQuantile=nullQuantile
    )

    numberOfClasses = len(clusterAssignments)
    classQuantiles = np.full(numberOfClasses, nullQuantile)
    validClusters = clusterAssignments >= 0
    classQuantiles[validClusters] = clusterQuantiles[clusterAssignments[validClusters]]

    return classQuantiles

def generatePredictionSets(validationScores, validationClusters, clusterQuantiles):
    """
    Generate prediction sets based on validation scores and cluster quantiles.
    """
    predictionSets = []
    for classScores, clusterId in zip(validationScores, validationClusters):
        quantileValue = clusterQuantiles.get(clusterId, jnp.inf)
        predictionSet = jnp.where(classScores <= quantileValue)[0]
        predictionSets.append(predictionSet)
    return predictionSets

def computeQhat(sampleScores, significanceLevel):
    """
    Compute the quantile q_hat for standard conformal prediction.
    """
    sampleCount = len(sampleScores)
    quantileIndex = int(jnp.ceil((sampleCount + 1) * (1 - significanceLevel)))
    quantileIndex = min(max(quantileIndex, 1), sampleCount)
    sortedScores = jnp.sort(sampleScores)
    quantileValue = sortedScores[quantileIndex - 1]
    return quantileValue



def quantile_embedding(samples, q=[0.5, 0.6, 0.7, 0.8, 0.9]):
    '''
    Computes the q-quantiles of samples and returns the vector of quantiles
    '''
    return np.quantile(samples, q)

def embed_all_classes(scores_all, labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=False):
    '''
    Input:
        - scores_all: num_instances x num_classes array where 
            scores_all[i,j] = score of class j for instance i
          Alternatively, num_instances-length array where scores_all[i] = score of true class for instance i
        - labels: num_instances-length array of true class labels
        - q: quantiles to include in embedding
        - return_cts: if True, return an array containing the counts for each class 
        
    Output: 
        - embeddings: num_classes x len(q) array where ith row is the embeddings of class i
        - (Optional) cts: num_classes-length array where cts[i] = # of times class i 
        appears in labels 
    '''
    num_classes = len(np.unique(labels))
    
    embeddings = np.zeros((num_classes, len(q)))
    cts = np.zeros((num_classes,))
    
    for i in range(num_classes):
        if len(scores_all.shape) == 2:
            class_i_scores = scores_all[labels==i,i]
        else:
            class_i_scores = scores_all[labels==i] 
        cts[i] = class_i_scores.shape[0]
        embeddings[i,:] = quantile_embedding(class_i_scores, q=q)
    
    if return_cts:
        return embeddings, cts
    else:
        return embeddings




    








