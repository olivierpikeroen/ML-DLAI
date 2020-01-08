import numpy as np 

def featureNormalize(X,mu=None,sigma=None):
    """
    Normalizes the features in X 
    
    Returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1. 
    This is often a good preprocessing step to do when working with learning algorithms.
    """
    if mu is None:
        mu=np.mean(X,axis=0)
    X_norm=X-mu
    if sigma is None:
        sigma=np.std(X_norm,axis=0)
    X_norm/=sigma
    return X_norm,mu,sigma

