import matplotlib.pyplot as plt 
import numpy as np 

def drawLine(p1, p2, **kwargs):
    """
    Draws a line from point p1 to point p2
    """
    plt.plot(np.linspace(p1[0], p2[0], 50), np.linspace(p1[1], p2[1], 50), **kwargs)
