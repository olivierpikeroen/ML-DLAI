import numpy as np 

def computeCost(X,y,theta):
    m=y.size
    D=X@theta-y
    return 1/(2*m)*D.T@D