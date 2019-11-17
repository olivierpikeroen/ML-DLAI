import numpy as np
import matplotlib.pyplot as plt
from plotData import *
from mapFeature import *

def plotDecisionBoundary(theta,X,y):
    plotData(X,y)
    # treshold: alpha=1/2 -> decision boundary: X@theta=0
    x=np.linspace(np.min(X[:,1]),np.max(X[:,1]),100)
    plt.plot(x,-1/theta[-1]*(theta[0]+theta[1]*x))

def plotDecisionBoundaryReg(theta,X,degree,**kwargs):
    #
    u=np.linspace(np.min(X[:,1]),np.max(X[:,1]),100)
    v=np.linspace(np.min(X[:,2]),np.max(X[:,2]),100)
    z=np.zeros((u.size,v.size))
    for i in range(u.size):
        for j in range(v.size):
            z[i,j]=mapFeature(u[i],v[j],degree)@theta
    cs=plt.contour(u,v,z,0,**kwargs)
    return cs
    