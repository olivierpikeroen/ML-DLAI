import numpy as np
from sigmoid import *

def predict(theta,X):
    theta=theta.reshape((theta.size,1))
    return sigmoid(X@theta),X@theta>=0

    