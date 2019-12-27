import numpy as np
from costFunction import *

def costFunctionReg(theta,X,y,reglambda):
    """
    Compute cost and gradient for logistic regression with regularization
    
    Computes the cost of using theta as the parameter for regularized logistic regression and the gradient of the cost w.r.t. to the parameters. 
    """
    return computeCostReg(theta,X,y,reglambda),computeGradReg(theta,X,y,reglambda)

def computeCostReg(theta,X,y,reglambda):
    """
    Compute cost for logistic regression with regularization
    
    Computes the cost of using theta as the parameter for regularized logistic regression. 
    """
    m=y.size
    J=computeCost(theta,X,y)+reglambda/(2*m)*np.sum(theta[1:]**2)
    return J

def computeGradReg(theta,X,y,reglambda):
    """
    Compute gradient for logistic regression with regularization
    
    Computes the gradient of the cost w.r.t. to the parameters for regularized logistic regression. 
    """
    m=y.size
    grad=computeGrad(theta,X,y)+reglambda/m*np.hstack((0,theta[1:]))
    return grad