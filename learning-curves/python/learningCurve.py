import numpy as np 
import matplotlib.pyplot as plt 
from trainLinearReg import *
from linearRegCostFunction import *

def learningCurve(X,y,Xval,yval,Lambda):
    """
    Generates the train and cross validation set errors needed to plot a learning curve
    
    Returns the train and cross validation set errors for a learning curve. In particular, it returns two vectors of the same length - error_train and error_val. 
    Then, error_train(i) contains the training error for i examples (and similarly for error_val(i)).
    In this function, you will compute the train and test errors for dataset sizes from 1 up to m. In practice, when working with larger datasets, you might want to do this in larger intervals.
    """
    m=X.shape[0]
    error_train=[]
    error_val=[]
    for i in range(1,m+1):
        theta=trainLinearReg(X[:i,:],y[:i],Lambda)
        error_train.append(linearRegCostFunction(X[:i,:],y[:i],theta,0)[0])
        error_val.append(linearRegCostFunction(Xval,yval,theta,0)[0])
    return np.array(error_train),np.array(error_val)

def plotLearningCurve(n_it,error_train,error_val):
    """
    Plot learning curve
    """
    plt.plot(range(n_it),error_train,error_val)
    plt.legend(('Train', 'Cross Validation'))
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')