import numpy as np 

def featureNormalize(X):
    """
    Normalizes the features in X 
    
    Returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1. 
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma
