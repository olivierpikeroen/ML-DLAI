import numpy as  np 
from trainLinearReg import *
from linearRegCostFunction import *

def validationCurve(X,y,Xval,yval):
    """
    Generate the train and validation errors needed to plot a validation curve that we can use to select lambda
    
    Returns the train and validation errors (in error_train, error_val) for different values of lambda. 
    You are given the training set (X,y) and validation set (Xval, yval).
    """
    lambda_vec=[0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
    error_train=[]
    error_val=[]
    for i in range(len(lambda_vec)):
        theta=trainLinearReg(X,y,lambda_vec[i])
        error_train.append(linearRegCostFunction(X,y,theta,0)[0])
        error_val.append(linearRegCostFunction(Xval,yval,theta,0)[0])
    return lambda_vec,error_train,error_val