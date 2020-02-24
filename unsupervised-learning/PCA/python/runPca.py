import numpy as np 

def runPca(X):
    """
    Run principal component analysis on the dataset X
    
    Computes eigenvectors of the covariance matrix of X
    Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """
    m = X.shape[0]
    Sigma = 1 / m * X.T @ X
    U, S, _ = np.linalg.svd(Sigma)
    return U, S