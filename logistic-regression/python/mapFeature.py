import numpy as np

def mapFeature(X1,X2,degree):
    """
    Feature mapping function to polynomial features

    Maps the two input features to quadratic features used in the regularization exercise.
    Returns a new feature array with more features, comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    Inputs X1, X2 must be the same size
    """
    map=np.ones_like(X1)
    for i in range(1,degree+1):
        for j in range(i+1):
            map=np.column_stack((map,X1**(i-j)*X2**j))
    return map
    