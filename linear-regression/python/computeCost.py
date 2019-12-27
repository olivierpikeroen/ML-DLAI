import numpy as np 

def computeCost(X,y,theta):
    """
    Compute cost for linear regression
    
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y
    """
    m=y.size
    D=X@theta-y
    return np.asscalar(1/(2*m)*D.T@D)