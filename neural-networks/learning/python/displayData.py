import numpy as np 
import matplotlib.image, matplotlib.pyplot as plt 

def displayData(X,example_height,example_width):
    """
    Display 2D data in a nice grid
    
    Displays 2D data stored in X in a nice grid. 
    It returns the figure handle h and the displayed array if requested.
    """
    m=X.shape[0]
    # Compute number of items to display
    n_row=int(np.floor(np.sqrt(m)))
    n_col=int(np.ceil(m/n_row))
    # Between images padding
    pad=1
    # Concatenate examples with paddings
    for i in range(n_row):
        # Make ith row
        for j in range(n_col):
            # Example index in X
            idx_example=i*n_col+j
            # First column
            if j==0:
                img_row=X[idx_example,:].reshape(example_height,example_width,order='F')
            # Other columns
            elif idx_example<m:
                img_row=np.concatenate((img_row,X[idx_example,:].reshape(example_height,example_width,order='F')),axis=1)
            # Empty example to complete the last row if not full
            else:
                img_row=np.concatenate((img_row,-np.ones((example_height,example_width))),axis=1)
            # Padding
            if j<n_col-1:
                img_row=np.concatenate((img_row,-np.ones((example_height,pad))),axis=1)
        # Concatenate rows
        # First row
        if i==0:
            img=img_row
        # Other rows
        else:
            img=np.concatenate((img,img_row),axis=0)
        # Padding
        if i<n_row-1: 
            img=np.concatenate((img,-np.ones((pad,img_row.shape[1]))),axis=0)
    # Plot
    _=plt.figure(figsize=(n_row,n_col))
    plt.imshow(img,cmap='gray')
    plt.axis('off')