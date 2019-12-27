import matplotlib.pyplot as plt 

def plotData(X,y,marker='b.',markersize=None,linewidth=None):
    """
    Plot training data
    """
    plt.plot(X,y,marker,markersize=markersize,linewidth=linewidth) 
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
