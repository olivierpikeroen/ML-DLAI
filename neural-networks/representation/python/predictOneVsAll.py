import numpy as np 
from sigmoid import *

def predictOneVsAll(all_theta,X):
    """
    Predict the label for a trained one-vs-all classifier. The labels are in the range 1..K, where K = size(all_theta, 1). 
  
    Will return a vector of predictions for each example in the matrix X. 
    Note that X contains the examples in rows. 
    all_theta is a matrix where the i-th row is a trained logistic regression theta vector for the i-th class. You should set p to a vector of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2 for 4 examples) 
    """
    m=X.shape[0]
    X=np.hstack((np.ones((m,1)),X))
    return np.argmax(sigmoid(X@all_theta.T),axis=1)+1