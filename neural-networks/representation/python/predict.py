import numpy as np 
from sigmoid import *

def predict(theta1,theta2,X):
    """
    Predict the label of an input given a trained neural network
    
    Outputs the predicted label of X given the trained weights of a neural network (Theta1, Theta2)
    """
    m=X.shape[0]
    X=np.hstack((np.ones((m,1)),X))
    a2=np.hstack((np.ones((m,1)),sigmoid(X@theta1.T)))
    return np.argmax(sigmoid(a2@theta2.T),axis=1)+1