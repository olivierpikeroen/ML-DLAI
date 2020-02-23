import numpy as np 

def computeCentroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the data points assigned to each centroid.
    
    Returns the new centroids by computing the means of the data points assigned to each centroid. 
    It is given a dataset X where each row is a single data point, 
    a vector idx of centroid assignments (i.e. each entry in range [1..K]) for each example, 
    and K, the number of centroids. 
    """
    return np.array([np.mean(X[np.array(idx) == k, :], axis=0) for k in range(1, K + 1)])