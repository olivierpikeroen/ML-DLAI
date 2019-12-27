import numpy as np 

def normalEqn(X,y):
    """
    Computes the closed-form solution to linear regression 

    Computes the closed-form solution to linear regression using the normal equations.
    """
    return np.linalg.pinv(X.T@X)@X.T@y