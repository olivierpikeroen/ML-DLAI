import numpy as np
import matplotlib.pyplot as plt 
from plotData import *

def visualizeBoundaryLinear(X, y, model):
    """
    Plots a linear decision boundary learned by the SVM

    Plots a linear decision boundary learned by the SVM and overlays the data on it
    """
    std = np.std(X, axis=0)
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - std[0] / 5, X[:, 0].max() + std[0] / 5, num=100), 
        np.linspace(X[:, 1].min() - std[1] / 5, X[:, 1].max() + std[1] / 5, num=100)
    )
    z = model.decision_function(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.contour(xx, yy, z, levels=1, colors='b')
    plotData(X, y)