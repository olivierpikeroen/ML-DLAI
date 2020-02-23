import numpy as np 

def findClosestCentroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Returns the closest centroids in idx for a dataset X where each row is a single example. 
    idx = m x 1 vector of centroid assignments (i.e. each entry in range [1..K])
    """
    return np.array([np.argmin(np.sum((X[i, :] - centroids) ** 2, axis=1)) + 1 for i in range(len(X))])