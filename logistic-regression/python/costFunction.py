import numpy as np
from sigmoid import *

def costFunction(theta,X,y):
    """
    Compute cost and gradient for logistic regression

    Computes the cost of using theta as the parameter for logistic regression and the gradient of the cost w.r.t. to the parameters.
    """
    return computeCost(theta,X,y),computeGrad(theta,X,y)

def computeCost(theta,X,y):
    """
    Compute cost for logistic regression

    Computes the cost of using theta as the parameter for logistic regression.
    """
    m=y.size
    theta=theta.reshape((theta.size,1))
    h=sigmoid(X@theta)
    J=-1/m*(y.T@np.log(h)+(1-y).T@np.log(1-h))
    return np.asscalar(J)

def computeGrad(theta,X,y):
    """
    Compute gradient for logistic regression

    Computes the gradient of the cost w.r.t. to the parameters for logistic regression.
    """
    m=y.size
    theta=theta.reshape((theta.size,1))
    h=sigmoid(X@theta)
    grad=1/m*X.T@(h-y)
    return grad.flatten()