import numpy as np 

def sigmoid(z):
    """
    Compute sigmoid function
    
    Computes the sigmoid of z.
    """
    return 1/(1+np.exp(-z))