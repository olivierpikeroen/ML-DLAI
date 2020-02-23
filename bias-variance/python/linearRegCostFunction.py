import numpy as np 

def linearRegCostFunction(X,y,theta,Lambda):
    """
    Compute cost and gradient for regularized linear regression with multiple variables
    
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y. 
    Returns the cost in J and the gradient in grad
    """
    m=len(X)
    theta=theta[:,np.newaxis]
    h=X@theta
    theta_reg=theta[1:]
    J=1/(2*m)*(np.sum((h-y)**2)+Lambda*np.sum(theta_reg**2))
    grad=1/m*(X.T@(h-y)+Lambda*np.vstack((0,theta_reg)))
    return np.asscalar(J),np.ravel(grad)