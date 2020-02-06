import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def gaussianKernelCV(X, y, Xval, yval):
    """
    Returns your choice of C and sigma learning parameters to use for SVM with RBF kernel
    
    Returns your choice of C and sigma. 
    """
    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    acc = np.zeros((len(C_vec), len(sigma_vec)))
    for i in range(len(C_vec)):
        for j in range(len(sigma_vec)):
            clf = SVC(C_vec[i], kernel='rbf', gamma=1 / (2 * sigma_vec[j] ** 2))
            clf.fit(X, y.ravel())
            acc[i, j] = clf.score(Xval, yval)
    ind = np.unravel_index(acc.argmax(), acc.shape)
    return C_vec[ind[0]], sigma_vec[ind[1]]