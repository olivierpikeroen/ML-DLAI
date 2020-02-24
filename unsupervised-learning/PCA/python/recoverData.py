def recoverData(Z, U, K):
    """
    Recovers an approximation of the original data when using the projected data

    Recovers an approximation the original data that has been reduced to K dimensions.
    It returns the approximate reconstruction.
    """
    return Z @ U[:, :K].T 