import numpy as np 
from sigmoid import *
from sigmoidGradient import *

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
    """
    Implements the neural network cost function for a two layer neural network which performs classification
    
    Computes the cost and gradient of the neural network. 
    The parameters for the neural network are "unrolled" into the vector nn_params and need to be converted back into the weight matrices. 
    The returned parameter grad should be a "unrolled" vector of the partial derivatives of the neural network.
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
    theta1=nn_params[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    theta2=nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)
    m=X.shape[0]
    X=np.hstack((np.ones((m,1)),X))
    h=sigmoid(np.hstack((np.ones((m,1)),sigmoid(X@theta1.T)))@theta2.T)
    J=0
    for k in range(1,num_labels+1):
        y_k=np.where(y==k,1,0)
        h_k=h[:,k-1]
        J-=1/m*(y_k.T@np.log(h_k)+(1-y_k).T@np.log(1-h_k))
    # Regularization
    theta1_nobias=theta1[:,1:] # 25x400
    theta2_nobias=theta2[:,1:] # 10x25
    J+=Lambda/(2*m)*(np.sum(theta1_nobias**2)+np.sum(theta2_nobias**2))
    # Backward propagation
    Delta1=np.zeros_like(theta1) # 25x401
    Delta2=np.zeros_like(theta2) # 10x26
    for t in range(m):
        a1=X[t,:][:,np.newaxis] # 401x1
        z2=theta1@a1 # 25x1
        a2=sigmoid(z2) # 25x1
        a3=sigmoid(theta2@np.vstack((1,a2))) # 10x1
        delta3=a3-(y[t]==np.arange(1,num_labels+1)[:,np.newaxis]) # 10x1
        delta2=theta2_nobias.T@delta3*sigmoidGradient(z2) # 25x1
        Delta1+=delta2@a1.T # 25x401
        Delta2+=delta3@np.vstack((1,a2)).T # 10x26
    theta1_grad=1/m*(Delta1+Lambda*np.hstack((np.zeros((theta1.shape[0],1)),theta1_nobias))) # 25x401
    theta2_grad=1/m*(Delta2+Lambda*np.hstack((np.zeros((theta2.shape[0],1)),theta2_nobias))) # 10x26
    return np.asscalar(J),np.append(np.ravel(theta1_grad),np.ravel(theta2_grad))
