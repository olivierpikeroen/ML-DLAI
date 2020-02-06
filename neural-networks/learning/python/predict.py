import numpy as np 
from sigmoid import *

def predict(theta1,theta2,X):
    """
    Predict the label of an input given a trained neural network
    
    Outputs the predicted label of X given the trained weights of a neural network (Theta1, Theta2)
    """
    m=X.shape[0]
    h1=sigmoid(np.hstack((np.ones((m,1)),X))@theta1.T)
    h2=sigmoid(np.hstack((np.ones((m,1)),h1))@theta2.T)
    return np.argmax(h2,axis=1)+1