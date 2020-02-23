import numpy as np 
from findClosestCentroids import *
from plotDataPoints import *
from plotProgresskMeans import *
from computeCentroids import *

def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    """
    runs the K-Means algorithm on data matrix X, where each row of X is a single example

    Runs the K-Means algorithm on data matrix X, where each row of X is a single example. 
    It uses initial_centroids used as the initial centroids. 
    max_iters specifies the total number of interactions of K-Means to execute. 
    plot_progress is a true/false flag that indicates if the function should also plot its progress as the learning happens. 
    This is set to false by default. 
    Returns centroids, a Kxn matrix of the computed centroids 
    and idx, a m x 1 vector of centroid assignments (i.e. each entry in range [1..K])
    """
    # Plot the data if we are plotting progress
    if plot_progress:
        _, ax = plt.subplots(max_iters, 1, figsize=(4, 4 * max_iters))
    # Initialize values
    K = initial_centroids.shape[0]
    centroid_history = [initial_centroids, initial_centroids]
    # Run K-Means
    for i in range(max_iters):
        # Output progress
        print('K-Means iteration %d/%d' %(i + 1, max_iters))
        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroid_history[-1])
        # Optionally, plot progress here
        if plot_progress:
            plt.sca(ax[i] if max_iters > 1 else ax)
            plt.title('Iteration number %d' % (i + 1))
            # Plot the examples
            plotDataPoints(X, idx)
            # Plot centroids progress
            for j in range(i + 1):
                plotProgresskMeans(centroid_history[j + 1], centroid_history[j])   
        # Given the memberships, compute new centroids
        centroid_history.append(computeCentroids(X, idx, K))
    return centroid_history[-1], idx  