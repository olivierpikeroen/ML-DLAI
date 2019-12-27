import numpy as np
from sigmoid import *

def predict(theta,X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters theta
    
    Computes the predictions for X using a threshold at 0.5 
    """
    theta=theta.reshape((theta.size,1))
    return sigmoid(X@theta),X@theta>=0

    