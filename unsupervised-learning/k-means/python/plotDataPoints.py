import matplotlib.pyplot as plt 

def plotDataPoints(X, idx):
    """
    Plots data points in X, coloring them so that those with the same index assignments in idx have the same color
    """
    colors = [{j: 'C' + str(j) for j in range(min(idx), max(idx) + 1)}[i] for i in idx]
    plt.scatter(X[:, 0], X[:, 1], s=15, c=colors)
