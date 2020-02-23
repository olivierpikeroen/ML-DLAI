import numpy as np 

def kMeansInitCentroids(X, K):
    """
    This function initializes K centroids that are to be used in K-Means on the dataset X

    Returns K initial centroids to be used with the K-Means on the dataset X
    """
    randidx = np.random.randint(X.shape[0], size=K)
    return X[randidx, :]