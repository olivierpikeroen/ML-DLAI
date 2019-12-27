import numpy as np 
from scipy import optimize
from lrCostFunction import *


def oneVsAll(X,y,n_labels,Lambda):
    """
    Trains multiple logistic regression classifiers and returns all the classifiers in a matrix all_theta, where the i-th row of all_theta corresponds to the classifier for label i
    
    Trains n_labels logistic regression classifiers and returns each of these classifiers in a matrix all_theta, where the i-th row of all_theta corresponds to the classifier for label i
    """
    m,n=X.shape
    X=np.hstack((np.ones((m,1)),X))
    all_theta=np.zeros((n_labels,n+1))
    for k in range(1,n_labels+1):
        res=optimize.minimize(lrCostFunction,np.ravel(all_theta[k-1,:]),args=(X,np.where(y==k,1,0),Lambda),method='BFGS',jac=True,options={'maxiter':50})
        all_theta[k-1,:]=res.x
    return all_theta