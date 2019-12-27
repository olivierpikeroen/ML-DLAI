import matplotlib.pyplot as plt 

def plotData(X,y):
    """
    Plots the data points X and y into a new figure 

    Plots the data points with + for the positive examples and o for the negative examples. X is assumed to be a Mx2 matrix.
    """
    line1=plt.plot(X[(y==1)[:,0],1],X[(y==1)[:,0],2],'+',linewidth=2,markersize=7)
    line2=plt.plot(X[(y==0)[:,0],1],X[(y==0)[:,0],2],'o',markersize=7)
    return line1,line2
    