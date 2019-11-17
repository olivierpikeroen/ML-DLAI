import numpy as np 
from computeCost import *

def gradientDescent(X,y,theta,alpha,num_iters):
    m=y.size
    J_history=np.zeros((num_iters,1))
    for iter in range(num_iters):
        theta-=alpha/m*X.T@(X@theta-y)
        J_history[iter]=computeCost(X,y,theta)
    return theta, J_history