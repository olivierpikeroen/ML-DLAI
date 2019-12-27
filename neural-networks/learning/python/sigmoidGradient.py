import numpy as np 
from sigmoid import *

def sigmoidGradient(z):
    """
    Returns the gradient of the sigmoid function evaluated at z
    
    Computes the gradient of the sigmoid function evaluated at z. 
    This should work regardless if z is a matrix or a vector. 
    In particular, if z is a vector or matrix, you should return the gradient for each element.
    """
    return sigmoid(z)*(1-sigmoid(z))