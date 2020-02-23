import numpy as np 
from scipy import optimize 
from linearRegCostFunction import *

def trainLinearReg(X,y,Lambda):
    """
    Trains linear regression given a dataset (X, y) and a regularization parameter lambda

    Trains linear regression using the dataset (X, y) and regularization parameter lambda. 
    Returns the trained parameters theta.
    """
    # Initialize Theta
    initial_theta=np.zeros((X.shape[1]))
    # Create "short hand" for the cost function to be minimized
    def costFunction(t): return linearRegCostFunction(X,y,t,Lambda)
    # Now, costFunction is a function that takes in only one argument
    options={
        'maxiter':200
    }
    # Minimize
    res=optimize.minimize(costFunction,initial_theta,jac=True,method='CG',options=options)
    return res.x 