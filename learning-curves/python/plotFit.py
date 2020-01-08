import numpy as np 
from polyFeatures import *
from plotData import *

def plotFit(X,y,mu,sigma,theta,p):
    """
    Plots a learned polynomial regression fit over an existing figure.
    Also works with linear regression.

    Plots the learned polynomial fit with power p and feature normalization (mu, sigma).
    """
    # We plot a range slightly bigger than the min and max values 
    # to get an idea of how the fit will vary outside the range of the data points
    x=np.arange(np.min(X)-15,np.max(X)+25,0.05)[:,np.newaxis]
    # Map the X values 
    X_poly,_,_=polyFeatureNormalize(x,p,mu=mu,sigma=sigma)
    # Add ones
    X_poly=np.hstack((np.ones((X_poly.shape[0],1)),X_poly))
    # Plot
    plotData(x,X_poly@theta,'--',linewidth=2)