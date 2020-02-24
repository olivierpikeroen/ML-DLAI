def projectData(X, U, K):
    """
    Computes the reduced data representation when projecting only on to the top k eigenvectors
    
    Computes the projection of the normalized inputs X into the reduced dimensional space spanned by the first K columns of U. 
    It returns the projected examples in Z.
    """
    return X @ U[:, :K]