import numpy as np
import matplotlib.pyplot as plt
from plotData import *
from mapFeature import *

def plotDecisionBoundary(theta,X,y,degree=None,**kwargs):
    """
    Plots the the decision boundary defined by theta into a new figure

    X is assumed to be a either 
    1) Mx3 matrix, where the first column is an all-ones column for the intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    """
    if X.shape[1]==3:
        # treshold: alpha=1/2 -> decision boundary: X@theta=0
        x=np.linspace(np.min(X[:,1]),np.max(X[:,1]),100)
        line=plt.plot(x,-1/theta[-1]*(theta[0]+theta[1]*x),**kwargs) 
    else:
        # Here is the grid range
        u=np.linspace(np.min(X[:,1]),np.max(X[:,1]),100)
        v=np.linspace(np.min(X[:,2]),np.max(X[:,2]),100)
        # Evaluate z = theta*x over the grid
        z=np.zeros((u.size,v.size))
        for i in range(u.size):
            for j in range(v.size):
                z[i,j]=mapFeature(u[i],v[j],degree)@theta
        # Plot z = 0
        line=plt.contour(u,v,z,0,**kwargs)
    return line

    
    