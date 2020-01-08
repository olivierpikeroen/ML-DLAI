import matplotlib.pyplot as plt 

def plotData(X,y,marker='b.',markersize=None,linewidth=None,**kwargs):
    """
    Plot training data
    """
    plt.plot(X,y,marker,markersize=markersize,linewidth=linewidth,**kwargs) 
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
