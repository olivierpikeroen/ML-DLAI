import numpy as np 
from debugInitializeWeights import *
from nnCostFunction import *
from computeNumericalGradient import *

def checkNNGradients(Lambda=0):
    """
    Creates a small neural network to check the backpropagation gradients
    
    Creates a small neural network to check the backpropagation gradients, it will output the analytical gradients produced by your backprop code and the numerical gradients (computed using computeNumericalGradient). 
    These two gradient computations should result in very similar values.
    """
    input_layer_size=3
    hidden_layer_size=5
    num_labels=3
    m=5
    # We generate some 'random' test data
    theta1=debugInitializeWeights(hidden_layer_size,input_layer_size)
    theta2=debugInitializeWeights(num_labels,hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X=debugInitializeWeights(m,input_layer_size-1)
    # Generate labels
    y=1+np.mod(np.arange(m)+1,num_labels)
    # Unroll parameters
    nn_params=np.append(np.ravel(theta1),np.ravel(theta2))
    # Short hand for cost function
    def costFunc(p): return nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda)
    _,grad=costFunc(nn_params)
    numgrad=computeNumericalGradient(costFunc,nn_params)
    # Visually examine the two gradient computations.  
    # The two columns should be very similar.
    print(np.hstack((numgrad[:,np.newaxis],grad[:,np.newaxis])))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)')
    # Evaluate the norm of the difference between the two solutions.  
    diff=np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If the backpropagation implementation is correct, then the relative difference will be small (less than 1e-9).'
    '\nRelative Difference: %g'%diff)