import numpy as np 
from sigmoid import *

def lrCostFunction(theta,X,y,Lambda):
    """
    Compute cost and gradient for logistic regression with regularization
    
    Computes the cost of using theta as the parameter for regularized logistic regression and the gradient of the cost w.r.t. to the parameters. 
    """
    m=len(y)
    theta=np.ravel(theta)[:,np.newaxis]
    h=sigmoid(X@theta)
    theta_reg=np.vstack((0,theta[1:]))
    J=-1/m*(y.T@np.log(h)+(1-y).T@np.log(1-h))+Lambda/(2*m)*np.sum(theta_reg**2)
    grad=1/m*(X.T@(h-y)+Lambda*theta_reg)
    return np.asscalar(J),np.ravel(grad)
