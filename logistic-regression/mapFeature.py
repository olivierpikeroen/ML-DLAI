import numpy as np

def mapFeature(X1,X2,degree):
    map=np.ones_like(X1)
    for i in range(1,degree+1):
        for j in range(i+1):
            map=np.column_stack((map,X1**(i-j)*X2**j))
    return map
    