import numpy as np
from costFunction import *

def costFunctionReg(theta,X,y,reglambda):
    return computeCostReg(theta,X,y,reglambda),computeGradReg(theta,X,y,reglambda)

def computeCostReg(theta,X,y,reglambda):
    m=y.size
    J=computeCost(theta,X,y)+reglambda/(2*m)*np.sum(theta[1:]**2)
    return J

def computeGradReg(theta,X,y,reglambda):
    m=y.size
    grad=computeGrad(theta,X,y)+reglambda/m*np.hstack((0,theta[1:]))
    return grad