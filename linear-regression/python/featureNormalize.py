import numpy as np 

def featureNormalize(X):
    """
    Normalizes the features in X 
    
    Returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1. 
    This is often a good preprocessing step to do when working with learning algorithms.
    """
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    return (X-mu)/sigma, mu, sigma