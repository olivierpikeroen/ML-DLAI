import numpy as np 
from featureNormalize import *

def polyFeatures(X,p):
    """
    Maps X (1D vector) into the p-th power
    
    Takes a data matrix X (size m x 1) and maps each example into its polynomial features where X_poly(i, :) = [X[i] X[i]**2 X[i]**3 ...  X[i]**p]
    """
    X_poly=np.zeros((X.size,p))
    for i in range(p):
        X_poly[:,i]=(X**(i+1)).flatten()
    return X_poly # X**(range(1,p+1))

def polyFeatureNormalize(X,p,mu=None,sigma=None):
    """
    Maps X (1D vector) into the p-th power and normalize result
    """
    return featureNormalize(polyFeatures(X,p),mu,sigma)