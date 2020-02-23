import matplotlib.pyplot as plt
from drawLine import *

def plotProgresskMeans(centroids, previous):
    """
    Helper function that displays the progress of k-Means as it is running. 
    It is intended for use only with 2D data.

    Plots a line between the previous locations and current locations of the centroids.
    """
    # Plot the centroids as black x's
    plt.plot(centroids[:, 0], centroids[:, 1], 'x', markeredgecolor='k', markersize=10, linewidth=3)
    # Plot the history of the centroids with lines
    for i in range(centroids.shape[0]):
        drawLine(centroids[i, :], previous[i, :], color='k')
